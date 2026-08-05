// Baseline ResNet-20 (CIFAR-10) on the open-source Cheddar release.
// Port of the AE workload (scale-snu/cheddar-ae unittest/ResNet.cpp) with the
// closed-source example_ops replaced by our reimplementation (ExampleOps.h).
//
// Level plan (v1, deterministic; see PORTING.md):
//   every ConvBN runs at level kConvLevel (consumes 1), EvalReLU manages its
//   own boot schedule, the pool/FC tail follows the AE code.

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../Testbed.h"
#include "DatasetUtils.h"
#include "ExampleOps.h"
#include "SignCoeffs.h"
#include "SignCoeffsHP.h"
#include "cnpy.h"

using word = uint32_t;
using namespace cheddar;
using namespace cheddar::example_ops;

using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;

static const double kReluRange =
    getenv("RELU_RANGE") ? atof(getenv("RELU_RANGE")) : 10.0;
static constexpr int kConvLevel = 2;
static constexpr int kPoolLevel = 2;
static constexpr int kFcLevel = 1;
static constexpr int kHalfDegree = 1 << 15;
static constexpr int kXWidth = 32;
static constexpr int warm_up = 0;

struct WeightPath {
  std::string weight_path;
  std::string bias_path;
};

static std::string Root(const std::string &rel) {
  // WEIGHTS_DIR: fused-npy 디렉토리 오버라이드 (예: 자체학습 C100 teacher).
  if (const char *e = getenv("WEIGHTS_DIR"))
    return std::string(e) + "/" + rel;
  return std::string(PROJECT_ROOT) + "/resnet20_fused/" + rel;
}

// path_list[i] = {conv1, conv2, (shortcut)} of block i (AE ordering), built
// at run time by probing the weight files so one driver serves ResNet-20
// (3 blocks/group) and ResNet-32 (5) — the 6n+2 family shares channels 16/32/64
// and therefore the packing; only the block count differs.
static std::vector<std::vector<WeightPath>> path_list;
static int blocks_per_group[3] = {0, 0, 0};

static bool FileExists(const std::string &path) {
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

static void BuildPathList() {
  path_list.clear();
  path_list.push_back(
      {{Root("conv1_reparam.weight"), Root("conv1_reparam.bias")}});
  const char *groups[3] = {"layer1", "layer2", "layer3"};
  for (int g = 0; g < 3; g++) {
    int n = 0;
    while (FileExists(Root(std::string(groups[g]) + "." + std::to_string(n) +
                           ".conv1_reparam.weight"))) {
      n++;
    }
    blocks_per_group[g] = n;
    for (int i = 0; i < n; i++) {
      std::string stem = std::string(groups[g]) + "." + std::to_string(i);
      std::vector<WeightPath> entry = {
          {Root(stem + ".conv1_reparam.weight"),
           Root(stem + ".conv1_reparam.bias")},
          {Root(stem + ".conv2_reparam.weight"),
           Root(stem + ".conv2_reparam.bias")}};
      if (FileExists(Root(stem + ".shortcut_reparam.weight"))) {
        entry.push_back({Root(stem + ".shortcut_reparam.weight"),
                         Root(stem + ".shortcut_reparam.bias")});
      }
      path_list.push_back(entry);
    }
  }
  path_list.push_back({{Root("linear.weight"), Root("linear.bias")}});
}

bool DirectoryExists(const std::string &directory) {
  struct stat buffer;
  return (stat(directory.c_str(), &buffer) == 0);
}

void DownloadCifar10Data() {
  std::string script = R"(
        #!/bin/bash
        wget -P cifar10_data https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
        tar -xvzf cifar10_data/cifar-10-binary.tar.gz -C cifar10_data
    )";
  std::ofstream file("cifar10_data/download.sh");
  file << script;
  file.close();
  std::string command =
      "chmod +x cifar10_data/download.sh && ./cifar10_data/download.sh";
  int rc = system(command.c_str());
  (void)rc;
}

void AddRequiredRotationsForTrace(EvkRequest &rotations, int start_rot_amount,
                                  int num_accum, int level) {
  for (int i = 1; i < num_accum; i *= 2) {
    int rot_amount = (start_rot_amount * i) % kHalfDegree;
    if (rot_amount < 0) rot_amount += kHalfDegree;
    rotations.AddRequest(rot_amount, level);
  }
}

// One BasicBlock: conv1 -> relu -> conv2 (+ shortcut) -> add -> relu.
class ResNetBlock {
 private:
  std::shared_ptr<BootContext<word>> context_;
  ConvBN<word> conv1_;
  ConvBN<word> conv2_;
  std::shared_ptr<EvalReLU<word>> relu_;
  std::unique_ptr<DownSample<word>> downsample_ = nullptr;

 public:
  ResNetBlock(std::shared_ptr<BootContext<word>> context,
              const TensorLayout &in, bool narrowing,
              std::shared_ptr<EvalReLU<word>> relu,
              std::vector<WeightPath> &paths)
      : context_{context},
        conv1_{context,
               in,
               narrowing ? in.channels * 2 : in.channels,
               3,
               narrowing ? 2 : 1,
               cnpy::npy_load(paths[0].weight_path).data<float>(),
               in.channels,
               cnpy::npy_load(paths[0].bias_path).data<float>(),
               1.0,
               1.0 / kReluRange,
               kConvLevel},
        conv2_{context,
               conv1_.out_,
               conv1_.out_.channels,
               3,
               1,
               cnpy::npy_load(paths[1].weight_path).data<float>(),
               conv1_.out_.channels,
               cnpy::npy_load(paths[1].bias_path).data<float>(),
               1.0,
               1.0 / kReluRange,
               kConvLevel},
        relu_{relu} {
    if (narrowing) {
      downsample_ = std::make_unique<DownSample<word>>(
          context, in, in.channels * 2,
          cnpy::npy_load(paths[2].weight_path).data<float>(),
          cnpy::npy_load(paths[2].bias_path).data<float>(), 1.0 / kReluRange,
          kConvLevel);
    }
  }

  const TensorLayout &OutLayout() const { return conv2_.out_; }

  void Evaluate(Ct &res, Ct &ct, const EvkMap<word> &evk_map) {
    Ct identity;
    context_->Copy(identity, ct);

    AdjustLevel(context_, ct, kConvLevel, evk_map);
    Ct tmp;
    conv1_.Evaluate(tmp, ct, evk_map);
    relu_->Evaluate(tmp, tmp, evk_map);
    AdjustLevel(context_, tmp, kConvLevel, evk_map);
    conv2_.Evaluate(tmp, tmp, evk_map);

    if (downsample_ != nullptr) {
      AdjustLevel(context_, identity, kConvLevel, evk_map);
      downsample_->Evaluate(res, identity, evk_map);
      context_->Add(res, res, tmp);
    } else {
      AdjustLevel(context_, identity, kConvLevel - 1, evk_map);
      context_->Add(res, tmp, identity);
    }
    relu_->Evaluate(res, res, evk_map);
  }

  void AddRequiredRotations(EvkRequest &rotations) {
    conv1_.AddRequiredRotations(rotations);
    conv2_.AddRequiredRotations(rotations);
    if (downsample_ != nullptr) downsample_->AddRequiredRotations(rotations);
  }
};

StripedMatrix ConvertToStripedMatrix(
    const std::vector<std::vector<Complex>> &a) {
  int height = a.size();
  int width = a[0].size();
  StripedMatrix c(height, width);
  for (int i = 0; i < width; i++) {
    std::vector<Complex> diag(height, Complex(0));
    bool all_zero = true;
    for (int j = 0; j < height; j++) {
      diag[j] = a[j][(i + j) % width];
      all_zero = all_zero && (diag[j] == Complex(0));
    }
    if (!all_zero) c.try_emplace(i, diag);
  }
  return c;
}

TEST_P(Testbed32, ResNet20) {
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr) << "resnetparam JSON must enable boot";

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(kNumSlots,
                                      BootVariant::kImaginaryRemoving);
  int end_level = boot_context->boot_param_.GetEndLevel();
  std::cout << "boot end level = " << end_level << std::endl;

  // SIGN_HP=1: MPCNN(α≈13)급 4단 고정밀 sign — "같은 환경에서의 MPCNN 정밀도 지점"
  // 실험용 (기본은 기존 3단 저정밀; memresnet 학생 회로는 이 스위치와 무관).
  const bool sign_hp = getenv("SIGN_HP") != nullptr;
  const auto &sign_stages = sign_hp ? kSignStagesHP : kSignStages;
  static const std::vector<bool> kNoCheb{};
  const auto &sign_cheb = sign_hp ? kSignStagesHPCheb : kNoCheb;
  std::cout << "[relu] sign stages = "
            << (sign_hp ? "HP v2 (15,15,27,27 · eps=2^-13 · cheb stages 2+)"
                        : "default (15,15,15 · eps=0.025)")
            << std::endl;
  auto relu = std::make_shared<EvalReLU<word>>(
      boot_context, kConvLevel - 1, end_level, sign_stages, sign_cheb);

  // ---- network construction ----------------------------------------------
  BuildPathList();
  ASSERT_GT(blocks_per_group[0], 0)
      << "no layer1.0 weights under " << Root("") << " (WEIGHTS_DIR wrong?)";
  const int total_blocks =
      blocks_per_group[0] + blocks_per_group[1] + blocks_per_group[2];
  std::cout << "blocks/group = " << blocks_per_group[0] << "/"
            << blocks_per_group[1] << "/" << blocks_per_group[2]
            << " -> ResNet-" << (2 * total_blocks + 2) << std::endl;
  TensorLayout input_layout{32, 1, 4};  // 3 real channels + 1 zero pad
  // conv0 folds the 1/relu_range normalization (weights AND bias).
  ConvBN<word> conv0(boot_context, input_layout, 16, 3, 1,
                     cnpy::npy_load(path_list[0][0].weight_path).data<float>(),
                     3, cnpy::npy_load(path_list[0][0].bias_path).data<float>(),
                     1.0 / kReluRange, 1.0 / kReluRange, kConvLevel);
  TensorLayout l1{32, 1, 16};
  // All BasicBlocks shallow→deep; narrowing (stride-2, channel×2) at the first
  // block of layer2/layer3. Layouts: l1 {32,1,16} → {16,2,32} → {8,4,64}.
  // LAZY_WEIGHTS=1: do not pre-encode all blocks' weight plaintexts (they are
  // device-resident and scale linearly with depth — ResNet-110 OOMs a 40 GB
  // A100). Instead the evaluation loop below encodes at most LAZY_CHUNK
  // blocks at a time and frees them after use, batching all images per chunk
  // so each block is still encoded exactly once per run. Encode+keygen time
  // is accumulated in keygen_us and reported for subtraction, mirroring the
  // existing phase-wise key residency.
  const bool lazy_weights = getenv("LAZY_WEIGHTS") != nullptr;
  std::vector<std::unique_ptr<ResNetBlock>> blocks;
  std::vector<int> block_group;
  std::vector<TensorLayout> block_in;
  std::vector<char> block_narrow;
  {
    TensorLayout cur = l1;
    for (int g = 0; g < 3; g++) {
      for (int i = 0; i < blocks_per_group[g]; i++) {
        bool narrowing = (g > 0 && i == 0);
        block_in.push_back(cur);
        block_narrow.push_back(narrowing);
        block_group.push_back(g);
        if (narrowing) {
          cur = TensorLayout{cur.width / 2, cur.pack * 2, cur.channels * 2};
        }
      }
    }
  }
  if (!lazy_weights) {
    for (size_t b = 0; b < block_in.size(); b++) {
      blocks.push_back(std::make_unique<ResNetBlock>(
          boot_context, block_in[b], block_narrow[b] != 0, relu,
          path_list[1 + b]));
      ASSERT_EQ(blocks.back()->OutLayout().channels,
                block_narrow[b] ? block_in[b].channels * 2
                                : block_in[b].channels);
    }
  }

  // ---- rotation key requests ---------------------------------------------
  // Phase-wise key residency: conv keys of one layer group are generated
  // right before the group and erased after it (~20 GB peak saving). Only
  // `rotations` (boot + pool + FC) stays resident for the whole run.
  EvkRequest rotations;
  EvkRequest group_req[3];
  conv0.AddRequiredRotations(group_req[0]);
  for (size_t b = 0; b < blocks.size(); b++) {
    blocks[b]->AddRequiredRotations(group_req[block_group[b]]);
  }
  boot_context->AddRequiredRotations(rotations, kNumSlots);

  // Key (re)generation is deployment setup, not per-image inference; phase-wise
  // residency forces it inside the loop, so time it separately (mirrors
  // MemResNet.cpp so the two are comparable).
  double keygen_us = 0;
  auto stopwatch = [&](auto &&fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    keygen_us += std::chrono::duration<double, std::micro>(
                     std::chrono::high_resolution_clock::now() - t0).count();
  };
  auto load_group_keys = [&](int g) {
    // NOTE: never skip on mere key existence — an existing key made for a
    // lower level has a too-small modulus and silently corrupts rotations.
    // The library call itself skips (with a warning) only when the existing
    // key's modulus is sufficient.
    stopwatch([&] { interface_->PrepareRotationKey(group_req[g]); });
  };
  auto drop_group_keys = [&](int g) {
    stopwatch([&] {
      for (const auto &[rot, level] : group_req[g]) {
        if (rotations.find(rot) != rotations.end()) continue;  // shared: keep
        bool used_later = false;
        for (int h = g + 1; h < 3 && !used_later; h++) {
          used_later = group_req[h].find(rot) != group_req[h].end();
        }
        if (!used_later) interface_->EraseRotationKey(rot);
      }
    });
  };

  // ---- avg-pool + FC tail (follows the AE public code) --------------------
  constexpr int pool_input_width = 8;
  constexpr int pool_pack = 4;
  constexpr int pool_channel = 64;
  int pool_z_channel = pool_channel / (pool_pack * pool_pack);

  PlainHoistMap pool_mask;
  pool_mask.try_emplace(0, std::map<int, Message>());
  for (int i = 0; i < pool_z_channel; i++) {
    for (int j = 0; j < pool_pack; j++) {
      int rot_src = (i * kXWidth * kXWidth) + j * kXWidth;
      int rot_dst = pool_pack * (i * pool_pack + j);
      int rot_idx = rot_src - rot_dst;
      if (rot_idx < 0) rot_idx += kHalfDegree;
      pool_mask[0].try_emplace(rot_idx, Message(kHalfDegree, Complex(0, 0)));
      auto &message = pool_mask[0][rot_idx];
      for (int l = 0; l < pool_pack; l++) {
        message[rot_dst + l] = Complex(
            1.0 / double(pool_input_width * pool_input_width) * kReluRange, 0);
      }
    }
  }
  HoistHandler<word> avg_pool(boot_context, pool_mask, kPoolLevel,
                              boot_context->param_.GetScale(kPoolLevel), false);
  avg_pool.AddRequiredRotations(rotations);
  constexpr int fc_feat = 64;
  cnpy::NpyArray fc_weight_npy =
      cnpy::npy_load(path_list.back()[0].weight_path);
  cnpy::NpyArray fc_bias_npy =
      cnpy::npy_load(path_list.back()[0].bias_path);
  const int ncls = static_cast<int>(fc_weight_npy.shape[0]);
  // 10 classes fit the natural 64 period; 100 classes use a 128 period (the
  // post-pool trace makes the layout 128-periodic with a zero upper half at no
  // extra level) with the weight matrix zero-padded to [ncls][128].
  const int fc_period = (ncls <= fc_feat) ? fc_feat : 2 * fc_feat;
  std::vector<float> fc_wpad;
  const float *fc_wp = fc_weight_npy.data<float>();
  if (fc_period != fc_feat) {
    fc_wpad.assign(static_cast<size_t>(ncls) * fc_period, 0.0f);
    for (int r = 0; r < ncls; r++)
      for (int c = 0; c < fc_feat; c++)
        fc_wpad[static_cast<size_t>(r) * fc_period + c] =
            fc_wp[static_cast<size_t>(r) * fc_feat + c];
    fc_wp = fc_wpad.data();
  }
  // Manual-BSGS dense layer (bias folded in); the library LinearTransform is
  // avoided for the same baby-step reason as the convs (see ExampleOps.h).
  ManualLinear<word> fc(boot_context, kHalfDegree, fc_period, ncls, fc_wp,
                        fc_bias_npy.data<float>(), kFcLevel);
  fc.AddRequiredRotations(rotations);

  AddRequiredRotationsForTrace(rotations, pool_pack, pool_input_width,
                               kPoolLevel);
  AddRequiredRotationsForTrace(rotations, kXWidth * pool_pack,
                               pool_input_width, kPoolLevel);
  AddRequiredRotationsForTrace(rotations, fc_period,
                               kHalfDegree / pool_input_width, kPoolLevel - 1);

  interface_->PrepareRotationKey(rotations);

  auto dbg = [&](const std::string &name, const Ct &ct) {
    if (!getenv("CHECK")) return;
    Ct snap;
    boot_context->Copy(snap, ct);
    std::vector<Complex> v;
    DecryptAndDecode(v, snap);
    double mx = 0;
    for (size_t j = 0; j < v.size(); j++) mx = std::max(mx, std::abs(v[j].real()));
    std::cout << "[dbg] " << name << " max|re|=" << mx << " first:";
    for (int j = 0; j < 6; j++) std::cout << " " << v[j].real();
    std::cout << std::endl;
  };

  // ---- data ---------------------------------------------------------------
  const bool is_c100 = (ncls == 100);
  std::string dataset_dir = is_c100 ? "cifar100_data" : "cifar10_data";
  if (is_c100) {
    if (!DirectoryExists(dataset_dir + "/cifar-100-binary")) {
      FAIL() << "cifar100_data/cifar-100-binary/test.bin required — generate "
                "with FHE-research/scripts/make_cifar100_bin.py";
    }
  } else if (!DirectoryExists(dataset_dir)) {
    mkdir(dataset_dir.c_str(), 0777);
    DownloadCifar10Data();
  }
  int num_test_images = 1;  // override with IMAGES=n
  if (const char *env = getenv("IMAGES")) num_test_images = atoi(env);
  int img_start = 0;  // override with IMG_START=n (window [n, n+IMAGES))
  if (const char *env = getenv("IMG_START")) img_start = atoi(env);
  CIFAR cifar("./" + dataset_dir, is_c100);
  cifar.read();
  cifar.transform({0, 0, 0}, {255, 255, 255});
  if (is_c100) {
    cifar.transform({0.5071, 0.4865, 0.4409}, {0.2673, 0.2564, 0.2762});
  } else {
    cifar.transform({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
  }
  Matrix_t test_data = cifar.test_data;
  Matrix_t test_labels = cifar.test_labels;
  Matrix_t output;
  output.resize(ncls, num_test_images);

  // input: 4 channel frames (3 real + zero pad), replicated to kNumSlots
  std::vector<Complex> input_vecs(kNumSlots, Complex(0, 0));
  std::vector<Complex> output_vec;

  if (lazy_weights) {
    // ---- layer-major batch evaluation, chunk-resident weights -------------
    int chunk_size = 6;
    if (const char *env = getenv("LAZY_CHUNK")) chunk_size = atoi(env);
    std::cout << "[lazy] weights chunk-resident, chunk = " << chunk_size
              << " blocks, batch = " << num_test_images << " images"
              << std::endl;
    std::vector<Ct> cts(num_test_images);
    __ProfileStart("ResNet20", warm_up, [&] {
      for (int i = 0; i < num_test_images; i++) {
        const int img = img_start + i;
        for (int rep = 0; rep < kNumSlots / (4 * 1024); rep++) {
          for (int j = 0; j < 3 * 1024; j++) {
            input_vecs[rep * 4 * 1024 + j] = Complex(test_data(j, img), 0.0);
          }
        }
        EncodeAndEncrypt(cts[i], input_vecs, kConvLevel);
      }
    }());
    std::cout << "-- Conv 0 --" << std::endl;
    load_group_keys(0);
    for (int i = 0; i < num_test_images; i++) {
      conv0.Evaluate(cts[i], cts[i], interface_->GetEvkMap());
      relu->Evaluate(cts[i], cts[i], interface_->GetEvkMap());
    }
    size_t b0 = 0;
    for (int g = 0; g < 3; g++) {
      std::cout << "-- Layer " << (g + 1) << " --" << std::endl;
      size_t gend = b0;
      while (gend < block_group.size() && block_group[gend] == g) gend++;
      for (size_t cs = b0; cs < gend; cs += chunk_size) {
        size_t ce = std::min(cs + static_cast<size_t>(chunk_size), gend);
        std::vector<std::unique_ptr<ResNetBlock>> chunk;
        stopwatch([&] {  // encode + incremental keygen = setup, subtracted
          for (size_t b = cs; b < ce; b++) {
            chunk.push_back(std::make_unique<ResNetBlock>(
                boot_context, block_in[b], block_narrow[b] != 0, relu,
                path_list[1 + b]));
            chunk.back()->AddRequiredRotations(group_req[g]);
          }
          interface_->PrepareRotationKey(group_req[g]);
        });
        for (int i = 0; i < num_test_images; i++) {
          for (auto &blk : chunk) {
            blk->Evaluate(cts[i], cts[i], interface_->GetEvkMap());
          }
        }
        stopwatch([&] { chunk.clear(); });  // free device plaintexts
      }
      drop_group_keys(g);
      b0 = gend;
    }
    std::cout << "-- AvgPool + FC --" << std::endl;
    for (int i = 0; i < num_test_images; i++) {
      AdjustLevel(boot_context, cts[i], kPoolLevel, interface_->GetEvkMap());
      cts[i].SetNumSlots(kHalfDegree);
      boot_context->Trace(cts[i], pool_pack, pool_input_width, cts[i],
                          interface_->GetEvkMap());
      boot_context->Trace(cts[i], kXWidth * pool_pack, pool_input_width,
                          cts[i], interface_->GetEvkMap());
      avg_pool.Evaluate(context_, cts[i], cts[i], interface_->GetEvkMap());
      boot_context->Trace(cts[i], fc_period, kHalfDegree / fc_period, cts[i],
                          interface_->GetEvkMap());
      AdjustLevel(boot_context, cts[i], kFcLevel, interface_->GetEvkMap());
      fc.Evaluate(cts[i], cts[i], interface_->GetEvkMap());
    }
    __ProfileEnd("ResNet20");
    std::cout << "[time] rotation-key (re)gen inside the timed region: "
              << static_cast<long>(keygen_us)
              << "us  <-- setup, subtract for inference-only cost"
              << std::endl;
    std::cout << "[lazy] wall/setup above cover the whole " << num_test_images
              << "-image batch — divide (wall - setup) by " << num_test_images
              << " for per-image time" << std::endl;
    keygen_us = 0;
    for (int i = 0; i < num_test_images; i++) {
      DecryptAndDecode(output_vec, cts[i]);
      std::cout << "logits[img " << img_start + i << "] (true label "
                << test_labels(img_start + i) << "): ";
      for (int j = 0; j < ncls; j++) {
        output(j, i) = output_vec[j].real();
        if (j < 10) std::cout << output_vec[j].real() << " ";
      }
      std::cout << (ncls > 10 ? "... (10/" + std::to_string(ncls) + ")" : "")
                << std::endl;
    }
  }

  Ct main_ct;
  for (int i = 0; lazy_weights ? false : i < num_test_images; i++) {
    const int img = img_start + i;
    for (int rep = 0; rep < kNumSlots / (4 * 1024); rep++) {
      for (int j = 0; j < 3 * 1024; j++) {
        input_vecs[rep * 4 * 1024 + j] = Complex(test_data(j, img), 0.0);
      }
    }
    __ProfileStart("ResNet20", warm_up,
                   EncodeAndEncrypt(main_ct, input_vecs, kConvLevel));
    // NOTE: phase-wise key generation currently sits inside the timed
    // region — fine for the correctness round, must be hoisted out (or
    // accounted separately) before any timing comparison.
    dbg("input", main_ct);
    std::cout << "-- Conv 0 --" << std::endl;
    load_group_keys(0);
    conv0.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("conv0", main_ct);
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("relu0", main_ct);
    std::cout << "-- Layer 1 --" << std::endl;
    int cur_g = 0, idx_in_group = 0;
    for (size_t b = 0; b < blocks.size(); b++) {
      if (block_group[b] != cur_g) {
        drop_group_keys(cur_g);
        cur_g = block_group[b];
        idx_in_group = 0;
        std::cout << "-- Layer " << (cur_g + 1) << " --" << std::endl;
        load_group_keys(cur_g);
      }
      blocks[b]->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
      dbg("block" + std::to_string(cur_g + 1) + "_" +
              std::to_string(++idx_in_group),
          main_ct);
    }
    drop_group_keys(2);

    std::cout << "-- AvgPool --" << std::endl;
    AdjustLevel(boot_context, main_ct, kPoolLevel, interface_->GetEvkMap());
    main_ct.SetNumSlots(kHalfDegree);
    boot_context->Trace(main_ct, pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    boot_context->Trace(main_ct, kXWidth * pool_pack, pool_input_width,
                        main_ct, interface_->GetEvkMap());
    avg_pool.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Trace(main_ct, fc_period, kHalfDegree / fc_period,
                        main_ct, interface_->GetEvkMap());
    dbg("pool", main_ct);

    std::cout << "-- FC --" << std::endl;
    AdjustLevel(boot_context, main_ct, kFcLevel, interface_->GetEvkMap());
    fc.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    __ProfileEnd("ResNet20");
    std::cout << "[time] rotation-key (re)gen inside the timed region: "
              << static_cast<long>(keygen_us) << "us  <-- setup, subtract for "
                                                 "inference-only cost"
              << std::endl;
    keygen_us = 0;

    DecryptAndDecode(output_vec, main_ct);
    std::cout << "logits[img " << img << "] (true label "
              << test_labels(img) << "): ";
    for (int j = 0; j < ncls; j++) {
      output(j, i) = output_vec[j].real();
      if (j < 10) std::cout << output_vec[j].real() << " ";
    }
    std::cout << (ncls > 10 ? "... (10/" + std::to_string(ncls) + ")" : "")
              << std::endl;
  }
  Matrix_t window_labels(num_test_images, 1);
  for (int i = 0; i < num_test_images; i++) {
    window_labels(i) = test_labels(img_start + i);
  }
  long double acc = compute_accuracy(output, window_labels);
  std::cout << "Accuracy (" << num_test_images << " images, from img "
            << img_start << "): " << acc << "  misses:";
  for (int i = 0; i < num_test_images; i++) {
    Matrix_t::Index arg;
    output.col(i).maxCoeff(&arg);
    if (int(arg) != int(window_labels(i))) std::cout << " " << img_start + i;
  }
  std::cout << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
