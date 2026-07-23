// Compressed ResNet-20 (MemoryBlock, memOFF) on the open-source Cheddar
// release — P4 of the Cheddar port. Reuses the verified example_ops (ConvBN /
// EvalPoly / EvalReLU / ManualLinear) via MemOps.h. Geometry + fused weights
// come from FHE-research/scripts/export_memresnet.py (manifest.json + *.npy);
// point MEMRESNET_DIR at that export directory.
//
// Circuit:  conv1(3->16,3x3) -> ReLU(sign) -> 9x MemOFFBlock -> avgpool -> fc
// The only sign-ReLU is after conv1; every block uses low-degree polynomials
// (no bootstrapping inside a block) -> far fewer boots than the baseline.
//
// Scale: single global S (manifest scale_S). Ciphertext carries value/S so
// boot inputs stay bounded; conv weights absorb S so polynomials see true
// scale (conv1 w,b *= 1/S; pool mask *= S to restore true scale for fc).

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../Testbed.h"
#include "DatasetUtils.h"
#include "ExampleOps.h"
#include "MemOps.h"
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
  int in_ch, out_ch, stride, rdown_k, b, rP_degree;
  bool shortcut;
};

static std::string Pref(const std::string &name) {
  std::string p = name;
  for (auto &c : p) if (c == '.') c = '_';
  return p;
}

void AddRequiredRotationsForTrace(EvkRequest &rotations, int start_rot_amount,
                                  int num_accum, int level) {
  for (int i = 1; i < num_accum; i *= 2) {
    int rot_amount = (start_rot_amount * i) % kHalfDegree;
    if (rot_amount < 0) rot_amount += kHalfDegree;
    rotations.AddRequest(rot_amount, level);
  }
}

bool DirExists(const std::string &d) {
  struct stat b;
  return stat(d.c_str(), &b) == 0;
}

TEST_P(Testbed32, MemResNet20) {
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
  std::cout << "MEMRESNET_DIR=" << dir << "  scale_S=" << S << std::endl;

  std::vector<BlockMeta> metas;
  for (const auto &b : J.at("blocks")) {
    BlockMeta m;
    m.name = b.at("name").get<std::string>();
    ASSERT_EQ(b.at("type").get<std::string>(), "memory")
        << "MemResNet v1 expects memoryblock export";
    ASSERT_FALSE(b.contains("P_degree"))
        << "memON export (has memory bank) not supported by MemResNet v1 — "
           "use a --no-memory (memOFF) checkpoint";
    m.in_ch = b.at("in_ch"); m.out_ch = b.at("out_ch"); m.stride = b.at("stride");
    m.rdown_k = b.at("rdown_k"); m.b = b.at("b"); m.rP_degree = b.at("rP_degree");
    m.shortcut = b.at("shortcut");
    std::cout << "  [meta] " << m.name << " in=" << m.in_ch << " out=" << m.out_ch
              << " s=" << m.stride << " k=" << m.rdown_k << " b=" << m.b
              << " rPdeg=" << m.rP_degree << " sc=" << m.shortcut << std::endl;
    ASSERT_GT(m.stride, 0);
    ASSERT_GT(m.b, 0);
    metas.push_back(m);
  }
  ASSERT_EQ(metas.size(), 9u) << "resnet20 has 9 blocks";

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

  // ---- 9 memOFF blocks (compiled at in_level = end_level) ----------------
  std::vector<std::unique_ptr<MemOFFBlock<word>>> blocks;
  TensorLayout cur{32, 1, 16};
  for (const auto &m : metas) {
    const std::string p = Pref(m.name);
    const float *sc = m.shortcut ? load(p + "__sc_w") : nullptr;
    auto coef = load_vecd(p + "__rP_coef");
    std::cout << "[build] " << m.name << " in{" << cur.width << "," << cur.pack
              << "," << cur.channels << "} rP_coef(" << coef.size() << "):";
    for (double c : coef) std::cout << " " << c;
    std::cout << std::flush;
    blocks.push_back(std::make_unique<MemOFFBlock<word>>(
        boot_context, cur, m.out_ch, m.stride, m.rdown_k, m.b,
        load(p + "__rdown_w"), load(p + "__rdown_b"),
        coef, load(p + "__rup_w"), sc, S, end_level));
    cur = blocks.back()->OutLayout();
    std::cout << " -> out{" << cur.width << "," << cur.pack << "," << cur.channels
              << "} level " << end_level << "->" << blocks.back()->OutLevel()
              << std::endl;
  }

  // ---- pool + fc tail (follows the baseline; kReluRange -> S) -------------
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

  constexpr int fc_input_width = 64;
  constexpr int fc_output_width = 10;
  cnpy::NpyArray fc_w = cnpy::npy_load(dir + "/fc_w.npy");
  cnpy::NpyArray fc_b = cnpy::npy_load(dir + "/fc_b.npy");
  ManualLinear<word> fc(boot_context, kHalfDegree, fc_input_width,
                        fc_output_width, fc_w.data<float>(), fc_b.data<float>(),
                        kFcLevel);
  std::cout << "[build] fc OK" << std::endl;

  // ---- rotation keys: resident (boot/pool/fc) + per-layer block groups ----
  EvkRequest rotations;
  EvkRequest group_req[3];
  conv1.AddRequiredRotations(group_req[0]);
  for (int i = 0; i < 9; i++) blocks[i]->AddRequiredRotations(group_req[i / 3]);
  boot_context->AddRequiredRotations(rotations, kNumSlots);
  avg_pool.AddRequiredRotations(rotations);
  AddRequiredRotationsForTrace(rotations, pool_pack, pool_input_width, kPoolLevel);
  AddRequiredRotationsForTrace(rotations, kXWidth * pool_pack, pool_input_width,
                               kPoolLevel);
  AddRequiredRotationsForTrace(rotations, pool_channel,
                               kHalfDegree / pool_input_width, kPoolLevel - 1);
  fc.AddRequiredRotations(rotations);
  std::cout << "[build] rotation requests: resident=" << rotations.size()
            << " g0=" << group_req[0].size() << " g1=" << group_req[1].size()
            << " g2=" << group_req[2].size() << std::endl;
  interface_->PrepareRotationKey(rotations);
  std::cout << "[build] resident keys OK" << std::endl;

  auto load_group_keys = [&](int g) { interface_->PrepareRotationKey(group_req[g]); };
  auto drop_group_keys = [&](int g) {
    for (const auto &[rot, level] : group_req[g]) {
      if (rotations.find(rot) != rotations.end()) continue;
      bool later = false;
      for (int h = g + 1; h < 3 && !later; h++)
        later = group_req[h].find(rot) != group_req[h].end();
      if (!later) interface_->EraseRotationKey(rot);
    }
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
  std::string dataset_dir = "cifar10_data";
  ASSERT_TRUE(DirExists(dataset_dir + "/cifar-10-batches-bin"))
      << "run the baseline resnet once to fetch " << dataset_dir;
  int num_test_images = 1;
  if (const char *e = getenv("IMAGES")) num_test_images = atoi(e);
  int img_start = 0;
  if (const char *e = getenv("IMG_START")) img_start = atoi(e);
  CIFAR cifar("./" + dataset_dir);
  cifar.read();
  cifar.transform({0, 0, 0}, {255, 255, 255});
  cifar.transform({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
  Matrix_t test_data = cifar.test_data, test_labels = cifar.test_labels;
  Matrix_t output; output.resize(10, num_test_images);

  std::vector<Complex> input_vecs(kNumSlots, Complex(0, 0)), output_vec;
  Ct main_ct;
  for (int i = 0; i < num_test_images; i++) {
    const int img = img_start + i;
    for (int rep = 0; rep < kNumSlots / (4 * 1024); rep++)
      for (int j = 0; j < 3 * 1024; j++)
        input_vecs[rep * 4 * 1024 + j] = Complex(test_data(j, img), 0.0);
    __ProfileStart("MemResNet20", warm_up,
                   EncodeAndEncrypt(main_ct, input_vecs, kConvLevel));
    dbg("input", main_ct);
    std::cout << "-- Conv 1 --" << std::endl;
    load_group_keys(0);
    conv1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("conv1", main_ct);
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    dbg("relu1", main_ct);
    int cur_group = 0;
    for (int bi = 0; bi < 9; bi++) {
      int g = bi / 3;
      if (g != cur_group) { drop_group_keys(cur_group); load_group_keys(g); cur_group = g; }
      AdjustLevel(boot_context, main_ct, end_level, interface_->GetEvkMap());
      blocks[bi]->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
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
    boot_context->Trace(main_ct, pool_channel, kHalfDegree / pool_channel,
                        main_ct, interface_->GetEvkMap());
    dbg("pool", main_ct);

    std::cout << "-- FC --" << std::endl;
    AdjustLevel(boot_context, main_ct, kFcLevel, interface_->GetEvkMap());
    fc.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    __ProfileEnd("MemResNet20");

    DecryptAndDecode(output_vec, main_ct);
    std::cout << "logits[img " << img << "] (true label " << test_labels(img)
              << "): ";
    for (int j = 0; j < 10; j++) { output(j, i) = output_vec[j].real();
      std::cout << output_vec[j].real() << " "; }
    std::cout << std::endl;
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
