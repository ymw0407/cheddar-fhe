// Isolation tests for the reimplemented example_ops (see PORTING.md §5).
// Narrows down the conv0 value explosion: scale path vs rotation path vs
// full 3x3 masks.

#include <cmath>
#include <vector>

#include "../Testbed.h"
#include "ExampleOps.h"
#include "SignCoeffs.h"
#include "cnpy.h"

using word = uint32_t;
using namespace cheddar;
using namespace cheddar::example_ops;

using Ct = Ciphertext<word>;

static constexpr int kLevel = 2;

// Encode a ramp pattern into layout {32,1,4}: value(c,y,x) = small ramp.
static void FillRamp(std::vector<Complex> &msg, const TensorLayout &lay) {
  msg.assign(kNumSlots, Complex(0, 0));
  int u = lay.UsedSlots();
  for (int c = 0; c < lay.channels; c++) {
    for (int y = 0; y < lay.width; y++) {
      for (int x = 0; x < lay.width; x++) {
        double v = 0.001 * (c + 1) + 0.0001 * y + 0.00001 * x;
        int s = lay.Slot(c, y, x);
        for (int rep = s; rep < kNumSlots; rep += u) msg[rep] = Complex(v, 0);
      }
    }
  }
}

TEST_P(Testbed32, IdentityConv1x1) {
  // c_out = c_in identity: every rotation is 0 --> pure pmult/scale path.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout lay{32, 1, 4};
  std::vector<float> w(4 * 4, 0.0f), b(4, 0.0f);
  for (int c = 0; c < 4; c++) w[c * 4 + c] = 1.0f;

  ConvBN<word> conv(boot_context, lay, 4, 1, 1, w.data(), 4, b.data(), 1.0,
                    1.0, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  FillRamp(msg, lay);
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  std::vector<Complex> out;
  DecryptAndDecode(out, ct);
  CompareMessages(msg, out, true, 1e-2);
}

TEST_P(Testbed32, ShiftConv1x1) {
  // c_out = c_in + 1 (mod 4): single frame-shift rotation --> key path.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout lay{32, 1, 4};
  std::vector<float> w(4 * 4, 0.0f), b(4, 0.0f);
  for (int co = 0; co < 4; co++) w[co * 4 + ((co + 1) % 4)] = 1.0f;

  ConvBN<word> conv(boot_context, lay, 4, 1, 1, w.data(), 4, b.data(), 1.0,
                    1.0, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  FillRamp(msg, lay);
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  // expected: out(c,y,x) = in(c+1 mod 4, y, x)
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u = lay.UsedSlots();
  for (int c = 0; c < 4; c++) {
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 32; x++) {
        int s = lay.Slot(c, y, x);
        Complex v = msg[lay.Slot((c + 1) % 4, y, x)];
        for (int rep = s; rep < kNumSlots; rep += u) expected[rep] = v;
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);
  CompareMessages(expected, out, true, 1e-2);
}

TEST_P(Testbed32, Conv0RealWeights) {
  // Full 3x3 conv0 against a CPU reference on the same ramp input.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout in{32, 1, 4};
  cnpy::NpyArray w_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.weight");
  cnpy::NpyArray b_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.bias");
  const float *w = w_npy.data<float>();
  const float *b = b_npy.data<float>();

  ConvBN<word> conv(boot_context, in, 16, 3, 1, w, 3, b, 0.1, 0.1, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  FillRamp(msg, in);
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  // CPU reference
  TensorLayout out_lay{32, 1, 16};
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u_out = out_lay.UsedSlots();
  for (int co = 0; co < 16; co++) {
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 32; x++) {
        double acc = 0.1 * b[co];
        for (int ci = 0; ci < 3; ci++) {
          for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
              int yy = y + di, xx = x + dj;
              if (yy < 0 || yy >= 32 || xx < 0 || xx >= 32) continue;
              double in_v = msg[in.Slot(ci, yy, xx)].real();
              double wv =
                  w[((static_cast<size_t>(co) * 3 + ci) * 3 + (di + 1)) * 3 +
                    (dj + 1)];
              acc += 0.1 * wv * in_v;
            }
          }
        }
        int s = out_lay.Slot(co, y, x);
        for (int rep = s; rep < kNumSlots; rep += u_out) {
          expected[rep] = Complex(acc, 0);
        }
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);
  CompareMessages(expected, out, true, 1e-2);
}

TEST_P(Testbed32, Conv0CifarMagnitude) {
  // Same conv0 but with CIFAR-scale inputs (|x| up to ~2.3) --> tests
  // whether the explosion is input-magnitude dependent (encoding overflow).
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout in{32, 1, 4};
  cnpy::NpyArray w_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.weight");
  cnpy::NpyArray b_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.bias");
  const float *w = w_npy.data<float>();
  const float *b = b_npy.data<float>();

  ConvBN<word> conv(boot_context, in, 16, 3, 1, w, 3, b, 0.1, 0.1, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  FillRamp(msg, in);
  for (auto &v : msg) v *= 300.0;  // scale ramp into CIFAR range
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  TensorLayout out_lay{32, 1, 16};
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u_out = out_lay.UsedSlots();
  for (int co = 0; co < 16; co++) {
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 32; x++) {
        double acc = 0.1 * b[co];
        for (int ci = 0; ci < 3; ci++) {
          for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
              int yy = y + di, xx = x + dj;
              if (yy < 0 || yy >= 32 || xx < 0 || xx >= 32) continue;
              double in_v = msg[in.Slot(ci, yy, xx)].real();
              double wv =
                  w[((static_cast<size_t>(co) * 3 + ci) * 3 + (di + 1)) * 3 +
                    (dj + 1)];
              acc += 0.1 * wv * in_v;
            }
          }
        }
        int s = out_lay.Slot(co, y, x);
        for (int rep = s; rep < kNumSlots; rep += u_out) {
          expected[rep] = Complex(acc, 0);
        }
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);
  CompareMessages(expected, out, true, 1e-2);
}

TEST_P(Testbed32, Conv0FullEnvironment) {
  // conv0 with the FULL ResNet environment set up first: boot prepared,
  // EvalReLU (3x EvalPoly compiled), extra keys --> tests environment
  // side effects on the conv path.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(kNumSlots,
                                      BootVariant::kImaginaryRemoving);
  int end_level = boot_context->boot_param_.GetEndLevel();
  EvalReLU<word> relu(boot_context, kLevel - 1, end_level,
                      kSignStages);

  TensorLayout in{32, 1, 4};
  cnpy::NpyArray w_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.weight");
  cnpy::NpyArray b_npy = cnpy::npy_load(
      std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.bias");
  const float *w = w_npy.data<float>();
  const float *b = b_npy.data<float>();

  ConvBN<word> conv(boot_context, in, 16, 3, 1, w, 3, b, 0.1, 0.1, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  boot_context->AddRequiredRotations(req, kNumSlots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  FillRamp(msg, in);
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  TensorLayout out_lay{32, 1, 16};
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u_out = out_lay.UsedSlots();
  for (int co = 0; co < 16; co++) {
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 32; x++) {
        double acc = 0.1 * b[co];
        for (int ci = 0; ci < 3; ci++) {
          for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
              int yy = y + di, xx = x + dj;
              if (yy < 0 || yy >= 32 || xx < 0 || xx >= 32) continue;
              double in_v = msg[in.Slot(ci, yy, xx)].real();
              double wv =
                  w[((static_cast<size_t>(co) * 3 + ci) * 3 + (di + 1)) * 3 +
                    (dj + 1)];
              acc += 0.1 * wv * in_v;
            }
          }
        }
        int s = out_lay.Slot(co, y, x);
        for (int rep = s; rep < kNumSlots; rep += u_out) {
          expected[rep] = Complex(acc, 0);
        }
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);
  CompareMessages(expected, out, true, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(
    CheddarOps, Testbed32, testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
