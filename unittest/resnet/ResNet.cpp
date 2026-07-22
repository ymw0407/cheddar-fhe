// Baseline ResNet-20 (CIFAR-10) on the open-source Cheddar release.
// Port of the AE workload (scale-snu/cheddar-ae unittest/ResNet.cpp) with the
// closed-source example_ops replaced by our reimplementation (ExampleOps.h).
//
// Level plan (v1, deterministic; see PORTING.md):
//   every ConvBN runs at level kConvLevel (consumes 1), EvalReLU manages its
//   own boot schedule, the pool/FC tail follows the AE code.

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../Testbed.h"
#include "DatasetUtils.h"
#include "ExampleOps.h"
#include "SignCoeffs.h"
#include "cnpy.h"

using word = uint32_t;
using namespace cheddar;
using namespace cheddar::example_ops;

using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;

static constexpr double kReluRange = 10.0;
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
  return std::string(PROJECT_ROOT) + "/resnet20_fused/" + rel;
}

// path_list[i] = {conv1, conv2, (shortcut)} of block i (AE ordering).
static std::vector<std::vector<WeightPath>> path_list = {
    {{Root("conv1_reparam.weight"), Root("conv1_reparam.bias")}},
    {{Root("layer1.0.conv1_reparam.weight"), Root("layer1.0.conv1_reparam.bias")},
     {Root("layer1.0.conv2_reparam.weight"), Root("layer1.0.conv2_reparam.bias")}},
    {{Root("layer1.1.conv1_reparam.weight"), Root("layer1.1.conv1_reparam.bias")},
     {Root("layer1.1.conv2_reparam.weight"), Root("layer1.1.conv2_reparam.bias")}},
    {{Root("layer1.2.conv1_reparam.weight"), Root("layer1.2.conv1_reparam.bias")},
     {Root("layer1.2.conv2_reparam.weight"), Root("layer1.2.conv2_reparam.bias")}},
    {{Root("layer2.0.conv1_reparam.weight"), Root("layer2.0.conv1_reparam.bias")},
     {Root("layer2.0.conv2_reparam.weight"), Root("layer2.0.conv2_reparam.bias")},
     {Root("layer2.0.shortcut_reparam.weight"), Root("layer2.0.shortcut_reparam.bias")}},
    {{Root("layer2.1.conv1_reparam.weight"), Root("layer2.1.conv1_reparam.bias")},
     {Root("layer2.1.conv2_reparam.weight"), Root("layer2.1.conv2_reparam.bias")}},
    {{Root("layer2.2.conv1_reparam.weight"), Root("layer2.2.conv1_reparam.bias")},
     {Root("layer2.2.conv2_reparam.weight"), Root("layer2.2.conv2_reparam.bias")}},
    {{Root("layer3.0.conv1_reparam.weight"), Root("layer3.0.conv1_reparam.bias")},
     {Root("layer3.0.conv2_reparam.weight"), Root("layer3.0.conv2_reparam.bias")},
     {Root("layer3.0.shortcut_reparam.weight"), Root("layer3.0.shortcut_reparam.bias")}},
    {{Root("layer3.1.conv1_reparam.weight"), Root("layer3.1.conv1_reparam.bias")},
     {Root("layer3.1.conv2_reparam.weight"), Root("layer3.1.conv2_reparam.bias")}},
    {{Root("layer3.2.conv1_reparam.weight"), Root("layer3.2.conv1_reparam.bias")},
     {Root("layer3.2.conv2_reparam.weight"), Root("layer3.2.conv2_reparam.bias")}},
    {{Root("linear.weight"), Root("linear.bias")}}};

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
               1.0,
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
               1.0,
               kConvLevel},
        relu_{relu} {
    if (narrowing) {
      downsample_ = std::make_unique<DownSample<word>>(
          context, in, in.channels * 2,
          cnpy::npy_load(paths[2].weight_path).data<float>(),
          cnpy::npy_load(paths[2].bias_path).data<float>(), kConvLevel);
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

  auto relu = std::make_shared<EvalReLU<word>>(boot_context, kConvLevel - 1,
                                               end_level, kSignStages);

  // ---- network construction ----------------------------------------------
  TensorLayout input_layout{32, 1, 4};  // 3 real channels + 1 zero pad
  // conv0 folds the 1/relu_range normalization (weights AND bias).
  ConvBN<word> conv0(boot_context, input_layout, 16, 3, 1,
                     cnpy::npy_load(path_list[0][0].weight_path).data<float>(),
                     3, cnpy::npy_load(path_list[0][0].bias_path).data<float>(),
                     1.0 / kReluRange, 1.0 / kReluRange, kConvLevel);
  TensorLayout l1{32, 1, 16};
  ResNetBlock block1_1(boot_context, l1, false, relu, path_list[1]);
  ResNetBlock block1_2(boot_context, l1, false, relu, path_list[2]);
  ResNetBlock block1_3(boot_context, l1, false, relu, path_list[3]);
  ResNetBlock block2_1(boot_context, l1, true, relu, path_list[4]);
  TensorLayout l2 = block2_1.OutLayout();  // {16, 2, 32}
  ResNetBlock block2_2(boot_context, l2, false, relu, path_list[5]);
  ResNetBlock block2_3(boot_context, l2, false, relu, path_list[6]);
  ResNetBlock block3_1(boot_context, l2, true, relu, path_list[7]);
  TensorLayout l3 = block3_1.OutLayout();  // {8, 4, 64}
  ResNetBlock block3_2(boot_context, l3, false, relu, path_list[8]);
  ResNetBlock block3_3(boot_context, l3, false, relu, path_list[9]);

  // ---- rotation key requests ---------------------------------------------
  EvkRequest rotations;
  conv0.AddRequiredRotations(rotations);
  block1_1.AddRequiredRotations(rotations);
  block1_2.AddRequiredRotations(rotations);
  block1_3.AddRequiredRotations(rotations);
  block2_1.AddRequiredRotations(rotations);
  block2_2.AddRequiredRotations(rotations);
  block2_3.AddRequiredRotations(rotations);
  block3_1.AddRequiredRotations(rotations);
  block3_2.AddRequiredRotations(rotations);
  block3_3.AddRequiredRotations(rotations);
  boot_context->AddRequiredRotations(rotations, kNumSlots);

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
                              boot_context->param_.GetScale(kPoolLevel), true);
  avg_pool.AddRequiredRotations(rotations);
  AddRequiredRotationsForTrace(rotations, pool_pack, pool_input_width,
                               kPoolLevel);
  AddRequiredRotationsForTrace(rotations, kXWidth * pool_pack,
                               pool_input_width, kPoolLevel);
  AddRequiredRotationsForTrace(rotations, pool_channel,
                               kHalfDegree / pool_input_width, kPoolLevel - 1);

  constexpr int fc_input_width = 64;
  constexpr int fc_output_width = 10;
  std::vector<std::vector<Complex>> fc_weight(
      fc_input_width, std::vector<Complex>(fc_input_width, 0.0));
  cnpy::NpyArray fc_weight_npy = cnpy::npy_load(path_list[10][0].weight_path);
  cnpy::NpyArray fc_bias_npy = cnpy::npy_load(path_list[10][0].bias_path);
  for (int i = 0; i < fc_output_width; i++) {
    for (int j = 0; j < fc_input_width; j++) {
      fc_weight[i][j] = fc_weight_npy.data<float>()[i * fc_input_width + j];
    }
  }
  LinearTransform<word> fc(boot_context, ConvertToStripedMatrix(fc_weight),
                           kFcLevel, boot_context->param_.GetScale(kFcLevel),
                           8, 8, 0, 0);
  fc.AddRequiredRotations(rotations);
  Pt fc_bias;
  std::vector<Complex> plain_fc_bias(fc_input_width, Complex(0, 0));
  for (int i = 0; i < fc_output_width; i++) {
    plain_fc_bias[i] = fc_bias_npy.data<float>()[i];
  }
  boot_context->encoder_.Encode(fc_bias, kFcLevel - 1,
                                boot_context->param_.GetScale(kFcLevel - 1),
                                plain_fc_bias);

  interface_->PrepareRotationKey(rotations);

  // ---- data ---------------------------------------------------------------
  std::string dataset_dir = "cifar10_data";
  if (!DirectoryExists(dataset_dir)) {
    mkdir(dataset_dir.c_str(), 0777);
    DownloadCifar10Data();
  }
  int num_test_images = 1;
  CIFAR cifar("./" + dataset_dir);
  cifar.read();
  cifar.transform({0, 0, 0}, {255, 255, 255});
  cifar.transform({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
  Matrix_t test_data = cifar.test_data;
  Matrix_t test_labels = cifar.test_labels;
  Matrix_t output;
  output.resize(10, num_test_images);

  // input: 4 channel frames (3 real + zero pad), replicated to kNumSlots
  std::vector<Complex> input_vecs(kNumSlots, Complex(0, 0));
  std::vector<Complex> output_vec;
  Ct main_ct;
  for (int i = 0; i < num_test_images; i++) {
    for (int rep = 0; rep < kNumSlots / (4 * 1024); rep++) {
      for (int j = 0; j < 3 * 1024; j++) {
        input_vecs[rep * 4 * 1024 + j] = Complex(test_data(j, i), 0.0);
      }
    }
    __ProfileStart("ResNet20", warm_up,
                   EncodeAndEncrypt(main_ct, input_vecs, kConvLevel));
    std::cout << "-- Conv 0 --" << std::endl;
    conv0.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Layer 1 --" << std::endl;
    block1_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block1_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block1_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Layer 2 --" << std::endl;
    block2_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block2_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block2_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Layer 3 --" << std::endl;
    block3_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block3_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block3_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());

    std::cout << "-- AvgPool --" << std::endl;
    AdjustLevel(boot_context, main_ct, kPoolLevel, interface_->GetEvkMap());
    main_ct.SetNumSlots(kHalfDegree);
    boot_context->Trace(main_ct, pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    boot_context->Trace(main_ct, kXWidth * pool_pack, pool_input_width,
                        main_ct, interface_->GetEvkMap());
    avg_pool.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Trace(main_ct, pool_channel, kHalfDegree / pool_channel,
                        main_ct, interface_->GetEvkMap());
    main_ct.SetNumSlots(fc_input_width);

    std::cout << "-- FC --" << std::endl;
    AdjustLevel(boot_context, main_ct, kFcLevel, interface_->GetEvkMap());
    fc.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Add(main_ct, main_ct, fc_bias);
    main_ct.SetNumSlots(kHalfDegree);
    __ProfileEnd("ResNet20");

    DecryptAndDecode(output_vec, main_ct);
    for (int j = 0; j < 10; j++) {
      output(j, i) = output_vec[j].real();
    }
  }
  long double acc = compute_accuracy(output, test_labels);
  std::cout << "Accuracy (" << num_test_images << " images): " << acc
            << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
