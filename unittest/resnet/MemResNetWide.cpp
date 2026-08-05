// Wide-b (multi-ciphertext) compressed ResNet on the open-source Cheddar
// release — 잔차 폭 b 를 슬롯 캡 너머로 키운 학생(b512/b1024)의 FHE 회로.
// 새 회로 실험은 공유 파일(MemResNet.cpp)을 건드리지 않는 규약에 따라 별도
// 파일 + 별도 타깃(memresnet_wide). 블록 구현은 MemOpsWide.h(MemBlockWide).
//
// MemResNet.cpp 와의 차이:
//   * 모든 "memory" 블록을 MemBlockWide 로 구성 — b <= cap 이면 조각 1개로
//     기존 MemBlock 과 동일 회로, b > cap 이면 ceil(b/cap) 조각 병렬.
//   * 메모리 뱅크 미지원 (b512/b1024 학생은 memOFF; manifest 에 P_degree 가
//     있으면 즉시 실패).
// 목적: 병목 해결책 ⓐ(잔차 폭)의 FHE 실측 — 정확도(평문 0.6356/0.6537 의
// 암호화 손실)와 시간(예측 0.109s x ct합: b512 ~0.44s ≈ 8.4x) 두 축.
//
// Scale: single global S (manifest scale_S) — MemResNet.cpp 와 동일 규약.

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../Testbed.h"
#include "DatasetUtils.h"
#include "ExampleOps.h"
#include "MemOpsWide.h"
#include "SignCoeffs.h"
#include "cnpy.h"

using word = uint32_t;
using namespace cheddar;
using namespace cheddar::example_ops;

using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;

static constexpr int kConvLevel = 2;
static constexpr int kPoolLevel = 2;
static constexpr int kFcLevel = 1;
static constexpr int kHalfDegree = 1 << 15;
static constexpr int kXWidth = 32;
static constexpr int warm_up = 0;

static std::string ExportDir() {
  if (const char *e = getenv("MEMRESNET_DIR")) return std::string(e);
  return std::string(PROJECT_ROOT) + "/memresnet_export";
}

// ---- block geometry parsed from manifest.json --------------------------------
struct BlockMeta {
  std::string name;   // e.g. "layer1.0"
  std::string type;   // "memory" (MemoryBlock) or "basic" (kept BasicBlock)
  int in_ch = 0, out_ch = 0, stride = 1, rdown_k = 3, b = 0, rP_degree = 2;
  bool shortcut = false;
  bool memory = false;   // bank term present (unsupported here)
  int n_bank = 0;
};

static std::string Pref(const std::string &name) {
  std::string p = name;
  for (auto &c : p) if (c == '.') c = '_';
  return p;
}

static void AddRequiredRotationsForTrace(EvkRequest &rotations,
                                         int start_rot_amount, int num_accum,
                                         int level) {
  for (int i = 1; i < num_accum; i *= 2) {
    int rot_amount = (start_rot_amount * i) % kHalfDegree;
    if (rot_amount < 0) rot_amount += kHalfDegree;
    rotations.AddRequest(rot_amount, level);
  }
}

static bool DirExists(const std::string &d) {
  struct stat b;
  return stat(d.c_str(), &b) == 0;
}

TEST_P(Testbed32, MemResNetWide) {
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr) << "resnetparam JSON must enable boot";

  const std::string dir = ExportDir();
  std::ifstream mf(dir + "/manifest.json");
  ASSERT_TRUE(mf.good()) << "manifest.json not found in " << dir;
  nlohmann::json J;
  mf >> J;
  const double S = J.at("scale_S").get<double>();
  std::cout << "MEMRESNET_DIR=" << dir << "  scale_S=" << S
            << "  [wide driver]" << std::endl;

  std::vector<BlockMeta> metas;
  for (const auto &b : J.at("blocks")) {
    BlockMeta m;
    m.name = b.at("name").get<std::string>();
    m.type = b.at("type").get<std::string>();
    m.in_ch = b.at("in_ch"); m.out_ch = b.at("out_ch"); m.stride = b.at("stride");
    m.shortcut = b.at("shortcut");
    if (m.type == "memory") {
      m.memory = b.contains("P_degree");
      if (m.memory) m.n_bank = b.at("N");
      m.rdown_k = b.at("rdown_k"); m.b = b.at("b"); m.rP_degree = b.at("rP_degree");
      ASSERT_GT(m.b, 0);
    }
    std::cout << "  [meta] " << m.name << " (" << m.type << ") in=" << m.in_ch
              << " out=" << m.out_ch << " s=" << m.stride << " sc=" << m.shortcut
              << (m.type == "basic" ? std::string("  [kept BasicBlock]")
                  : "  b=" + std::to_string(m.b))
              << std::endl;
    ASSERT_FALSE(m.memory)
        << m.name << ": wide driver 는 메모리 뱅크 미지원 — memOFF export 사용";
    ASSERT_GT(m.stride, 0);
    metas.push_back(m);
  }
  ASSERT_GE(metas.size(), 3u);

  // keep all loaded npy alive for the whole setup (ConvBN copies at ctor)
  std::map<std::string, cnpy::NpyArray> npy;
  auto load = [&](const std::string &f) -> const float * {
    auto it = npy.find(f);
    if (it == npy.end())
      it = npy.emplace(f, cnpy::npy_load(dir + "/" + f + ".npy")).first;
    return it->second.data<float>();
  };
  auto load_vecd = [&](const std::string &f) {
    auto a = cnpy::npy_load(dir + "/" + f + ".npy");
    const float *p = a.data<float>();
    return std::vector<double>(p, p + a.num_vals);
  };

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(kNumSlots,
                                      BootVariant::kImaginaryRemoving);
  const int end_level = boot_context->boot_param_.GetEndLevel();
  std::cout << "boot end level = " << end_level << std::endl;

  // ---- conv1 + relu (the only sign-ReLU) ---------------------------------
  TensorLayout input_layout{32, 1, 4};  // 3 real + 1 zero pad
  ConvBN<word> conv1(boot_context, input_layout, 16, 3, 1,
                     load("conv1_w"), 3, load("conv1_b"),
                     1.0 / S, 1.0 / S, kConvLevel);
  std::cout << "[build] conv1 OK" << std::endl;
  auto relu = std::make_shared<EvalReLU<word>>(boot_context, kConvLevel - 1,
                                               end_level, kSignStages);
  std::cout << "[build] relu OK (out level " << relu->OutputLevel() << ")"
            << std::endl;

  // ---- compressed blocks (MemBlockWide) + kept BasicBlocks ----------------
  struct AnyBlock {
    std::unique_ptr<MemBlockWide<word>> mem;
    std::unique_ptr<ConvBN<word>> c1, c2, sc;  // kept BasicBlock parts
    TensorLayout out;
  };
  std::vector<AnyBlock> blocks;
  TensorLayout cur{32, 1, 16};
  for (const auto &m : metas) {
    const std::string p = Pref(m.name);
    AnyBlock ab;
    if (m.type == "basic") {
      TensorLayout mid{cur.width / m.stride, cur.pack * m.stride, m.out_ch};
      ab.c1 = std::make_unique<ConvBN<word>>(
          boot_context, cur, m.out_ch, 3, m.stride, load(p + "__conv1_w"),
          cur.channels, load(p + "__conv1_b"), 1.0, 1.0 / S, kConvLevel);
      ab.c2 = std::make_unique<ConvBN<word>>(
          boot_context, mid, m.out_ch, 3, 1, load(p + "__conv2_w"),
          m.out_ch, load(p + "__conv2_b"), 1.0, 1.0 / S, kConvLevel);
      if (m.shortcut) {
        ab.sc = std::make_unique<ConvBN<word>>(
            boot_context, cur, m.out_ch, 1, m.stride, load(p + "__sc_w"),
            cur.channels, load(p + "__sc_b"), 1.0, 1.0 / S, kConvLevel);
      }
      ab.out = mid;
      std::cout << "[build] " << m.name << " kept BasicBlock in{" << cur.width
                << "," << cur.pack << "," << cur.channels << "}" << std::endl;
    } else {
      const float *sc = m.shortcut ? load(p + "__sc_w") : nullptr;
      auto coef = load_vecd(p + "__rP_coef");
      std::cout << "[build] " << m.name << " in{" << cur.width << "," << cur.pack
                << "," << cur.channels << "} rP_coef(" << coef.size() << "):";
      for (double c : coef) std::cout << " " << c;
      std::cout << std::flush;
      ab.mem = std::make_unique<MemBlockWide<word>>(
          boot_context, cur, m.out_ch, m.stride, m.rdown_k, m.b,
          load(p + "__rdown_w"), load(p + "__rdown_b"),
          coef, load(p + "__rup_w"), sc, S, end_level);
      ab.out = ab.mem->OutLayout();
      std::cout << " -> level " << end_level << "->" << ab.mem->OutLevel()
                << std::endl;
    }
    cur = ab.out;
    blocks.push_back(std::move(ab));
  }

  // ---- pool + fc tail (MemResNet.cpp 와 동일) -----------------------------
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
      for (int l = 0; l < pool_pack; l++)
        message[rot_dst + l] =
            Complex(1.0 / double(pool_input_width * pool_input_width) * S, 0);
    }
  }
  HoistHandler<word> avg_pool(boot_context, pool_mask, kPoolLevel,
                              boot_context->param_.GetScale(kPoolLevel), false);
  std::cout << "[build] pool OK" << std::endl;

  constexpr int fc_feat = 64;
  cnpy::NpyArray fc_w = cnpy::npy_load(dir + "/fc_w.npy");
  cnpy::NpyArray fc_b = cnpy::npy_load(dir + "/fc_b.npy");
  const int ncls = static_cast<int>(fc_w.shape[0]);
  const int fc_period = (ncls <= fc_feat) ? fc_feat : 2 * fc_feat;
  std::vector<float> fc_wpad;
  const float *fc_wp = fc_w.data<float>();
  if (fc_period != fc_feat) {
    fc_wpad.assign(static_cast<size_t>(ncls) * fc_period, 0.0f);
    for (int r = 0; r < ncls; r++)
      for (int c = 0; c < fc_feat; c++)
        fc_wpad[static_cast<size_t>(r) * fc_period + c] =
            fc_wp[static_cast<size_t>(r) * fc_feat + c];
    fc_wp = fc_wpad.data();
  }
  ManualLinear<word> fc(boot_context, kHalfDegree, fc_period, ncls, fc_wp,
                        fc_b.data<float>(), kFcLevel);
  std::cout << "[build] fc OK (ncls=" << ncls << " period=" << fc_period << ")"
            << std::endl;

  const bool is_c100 = (ncls == 100);
  const std::string dataset_dir = is_c100 ? "cifar100_data" : "cifar10_data";
  if (is_c100) {
    ASSERT_TRUE(DirExists(dataset_dir + "/cifar-100-binary"))
        << "cifar100_data/cifar-100-binary/test.bin required — generate with "
           "FHE-research/scripts/make_cifar100_bin.py";
  } else {
    ASSERT_TRUE(DirExists(dataset_dir + "/cifar-10-batches-bin"))
        << "run the baseline resnet once to fetch " << dataset_dir;
  }

  // ---- rotation keys: resident (boot/pool/fc) + per-layer block groups ----
  EvkRequest rotations;
  static constexpr int kMaxGroups = 4;
  EvkRequest group_req[kMaxGroups];
  conv1.AddRequiredRotations(group_req[0]);
  auto layer_group = [](const std::string &name) {
    const int g = name[5] - '1';
    return (g >= 0 && g < kMaxGroups) ? g : 0;
  };
  for (size_t i = 0; i < blocks.size(); i++) {
    int g = layer_group(metas[i].name);
    if (blocks[i].mem) {
      blocks[i].mem->AddRequiredRotations(group_req[g]);
    } else {
      blocks[i].c1->AddRequiredRotations(group_req[g]);
      blocks[i].c2->AddRequiredRotations(group_req[g]);
      if (blocks[i].sc) blocks[i].sc->AddRequiredRotations(group_req[g]);
    }
  }
  boot_context->AddRequiredRotations(rotations, kNumSlots);
  avg_pool.AddRequiredRotations(rotations);
  AddRequiredRotationsForTrace(rotations, pool_pack, pool_input_width, kPoolLevel);
  AddRequiredRotationsForTrace(rotations, kXWidth * pool_pack, pool_input_width,
                               kPoolLevel);
  AddRequiredRotationsForTrace(rotations, fc_period,
                               kHalfDegree / pool_input_width, kPoolLevel - 1);
  fc.AddRequiredRotations(rotations);
  std::cout << "[build] rotation requests: resident=" << rotations.size()
            << " g0=" << group_req[0].size() << " g1=" << group_req[1].size()
            << " g2=" << group_req[2].size() << std::endl;
  interface_->PrepareRotationKey(rotations);
  std::cout << "[build] resident keys OK" << std::endl;

  double keygen_us = 0;
  auto stopwatch = [&](auto &&fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    keygen_us += std::chrono::duration<double, std::micro>(
                     std::chrono::high_resolution_clock::now() - t0).count();
  };
  auto load_group_keys = [&](int g) {
    stopwatch([&] { interface_->PrepareRotationKey(group_req[g]); });
  };
  auto drop_group_keys = [&](int g) {
    stopwatch([&] {
      for (const auto &[rot, level] : group_req[g]) {
        if (rotations.find(rot) != rotations.end()) continue;
        bool later = false;
        for (int h = g + 1; h < kMaxGroups && !later; h++)
          later = group_req[h].find(rot) != group_req[h].end();
        if (!later) interface_->EraseRotationKey(rot);
      }
    });
  };

  auto dbg = [&](const std::string &name, const Ct &ct) {
    if (!getenv("CHECK")) return;
    Ct snap; boot_context->Copy(snap, ct);
    std::vector<Complex> v; DecryptAndDecode(v, snap);
    double mx = 0;
    for (auto &z : v) mx = std::max(mx, std::abs(z.real()));
    std::cout << "[dbg] " << name << " max|re|=" << mx << " first:";
    for (int j = 0; j < 6; j++) std::cout << " " << v[j].real();
    std::cout << std::endl;
  };

  // ---- data --------------------------------------------------------------
  int num_test_images = 1;
  if (const char *e = getenv("IMAGES")) num_test_images = atoi(e);
  int img_start = 0;
  if (const char *e = getenv("IMG_START")) img_start = atoi(e);
  CIFAR cifar("./" + dataset_dir, is_c100);
  cifar.read();
  cifar.transform({0, 0, 0}, {255, 255, 255});
  if (is_c100) {
    cifar.transform({0.5071, 0.4865, 0.4409}, {0.2673, 0.2564, 0.2762});
  } else {
    cifar.transform({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
  }
  Matrix_t test_data = cifar.test_data, test_labels = cifar.test_labels;
  Matrix_t output; output.resize(ncls, num_test_images);

  std::vector<Complex> input_vecs(kNumSlots, Complex(0, 0)), output_vec;
  Ct main_ct;
  for (int i = 0; i < num_test_images; i++) {
    const int img = img_start + i;
    for (int rep = 0; rep < kNumSlots / (4 * 1024); rep++)
      for (int j = 0; j < 3 * 1024; j++)
        input_vecs[rep * 4 * 1024 + j] = Complex(test_data(j, img), 0.0);
    __ProfileStart("MemResNetWide", warm_up,
                   EncodeAndEncrypt(main_ct, input_vecs, kConvLevel));
    dbg("input", main_ct);
    std::cout << "-- Conv 1 --" << std::endl;
    load_group_keys(0);
    conv1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("conv1", main_ct);
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("relu1", main_ct);
    int cur_group = 0;
    for (size_t bi = 0; bi < blocks.size(); bi++) {
      int g = layer_group(metas[bi].name);
      if (g != cur_group) { drop_group_keys(cur_group); load_group_keys(g); cur_group = g; }
      if (blocks[bi].mem) {
        AdjustLevel(boot_context, main_ct, end_level, interface_->GetEvkMap());
        blocks[bi].mem->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
      } else {
        Ct identity;
        boot_context->Copy(identity, main_ct);
        AdjustLevel(boot_context, main_ct, kConvLevel, interface_->GetEvkMap());
        Ct tmp;
        blocks[bi].c1->Evaluate(tmp, main_ct, interface_->GetEvkMap());
        relu->Evaluate(tmp, tmp, interface_->GetEvkMap());
        AdjustLevel(boot_context, tmp, kConvLevel, interface_->GetEvkMap());
        blocks[bi].c2->Evaluate(tmp, tmp, interface_->GetEvkMap());
        if (blocks[bi].sc) {
          AdjustLevel(boot_context, identity, kConvLevel, interface_->GetEvkMap());
          blocks[bi].sc->Evaluate(main_ct, identity, interface_->GetEvkMap());
          boot_context->Add(main_ct, main_ct, tmp);
        } else {
          AdjustLevel(boot_context, identity, kConvLevel - 1,
                      interface_->GetEvkMap());
          boot_context->Add(main_ct, tmp, identity);
        }
        relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
      }
      dbg(metas[bi].name, main_ct);
    }
    drop_group_keys(cur_group);

    std::cout << "-- AvgPool --" << std::endl;
    AdjustLevel(boot_context, main_ct, kPoolLevel, interface_->GetEvkMap());
    main_ct.SetNumSlots(kHalfDegree);
    boot_context->Trace(main_ct, pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    boot_context->Trace(main_ct, kXWidth * pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    avg_pool.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Trace(main_ct, fc_period, kHalfDegree / fc_period,
                        main_ct, interface_->GetEvkMap());
    dbg("pool", main_ct);

    std::cout << "-- FC --" << std::endl;
    AdjustLevel(boot_context, main_ct, kFcLevel, interface_->GetEvkMap());
    fc.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    __ProfileEnd("MemResNetWide");
    std::cout << "[time] rotation-key (re)gen inside the timed region: "
              << static_cast<long>(keygen_us) << "us  <-- setup, subtract for "
                                                 "inference-only cost"
              << std::endl;
    keygen_us = 0;

    DecryptAndDecode(output_vec, main_ct);
    std::cout << "logits[img " << img << "] (true label " << test_labels(img)
              << "): ";
    for (int j = 0; j < ncls; j++) {
      output(j, i) = output_vec[j].real();
      if (j < 10) std::cout << output_vec[j].real() << " ";
    }
    std::cout << (ncls > 10 ? "... (10/" + std::to_string(ncls) + ")" : "")
              << std::endl;
  }
  Matrix_t win(num_test_images, 1);
  for (int i = 0; i < num_test_images; i++) win(i) = test_labels(img_start + i);
  long double acc = compute_accuracy(output, win);
  std::cout << "Accuracy (" << num_test_images << " images, from img "
            << img_start << "): " << acc << "  misses:";
  for (int i = 0; i < num_test_images; i++) {
    Matrix_t::Index arg; output.col(i).maxCoeff(&arg);
    if (int(arg) != int(win(i))) std::cout << " " << img_start + i;
  }
  std::cout << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
