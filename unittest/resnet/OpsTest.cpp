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

TEST_P(Testbed32, DeltaTapGeometry) {
  // Delta input at (c=0, y=8, x=8); single-output 3x3 kernel with distinct
  // tap values --> the decrypted stamp reveals the exact tap mapping.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout in{32, 1, 4};
  std::vector<float> w(1 * 4 * 3 * 3, 0.0f), b(1, 0.0f);
  for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
      w[(0 * 4 + 0) * 9 + di * 3 + dj] = 1.0f + di * 3 + dj;  // 1..9

  ConvBN<word> conv(boot_context, in, 1, 3, 1, w.data(), 4, b.data(), 1.0,
                    1.0, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg(kNumSlots, Complex(0, 0));
  int u_in = in.UsedSlots();
  for (int rep = in.Slot(0, 8, 8); rep < kNumSlots; rep += u_in) {
    msg[rep] = Complex(1.0, 0);
  }
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  // expected stamp: out(y,x) = w[y-8+1][x-8+1] for taps hitting the delta:
  // out(8+di', 8+dj') gets tap (di=-di'? ...) -- direct formula:
  // out(y,x) = sum_taps w[di][dj] * delta(y+di-1... ) with our convention
  // out(y,x) uses input(y+di, x+dj), di,dj in {-1,0,1} => delta at (8,8)
  // contributes to out(8-di, 8-dj) with weight w[di+1][dj+1].
  TensorLayout out_lay{32, 1, 1};
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u_out = out_lay.UsedSlots();
  for (int di = -1; di <= 1; di++) {
    for (int dj = -1; dj <= 1; dj++) {
      int y = 8 - di, x = 8 - dj;
      double wv = w[(di + 1) * 3 + (dj + 1)];
      for (int rep = out_lay.Slot(0, y, x); rep < kNumSlots; rep += u_out) {
        expected[rep] = Complex(wv, 0);
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);

  // dump EVERY slot with significant mass (full ring) --> reveals the
  // library's actual rotation convention: slot -> (frame, y, x)
  int shown = 0;
  for (int s = 0; s < kNumSlots && shown < 40; s++) {
    double got = out[s].real(), exp = expected[s].real();
    if (std::abs(got) > 1e-3 || std::abs(exp) > 1e-3) {
      int f = s / kFrame, rem = s % kFrame;
      std::cout << "slot " << s << " (f=" << f << ",y=" << rem / kEdge
                << ",x=" << rem % kEdge << ") expected=" << exp
                << " got=" << got << std::endl;
      shown++;
    }
  }
  int bad = 0;
  for (int s = 0; s < kNumSlots; s++) {
    if (std::abs(out[s].real() - expected[s].real()) > 1e-3) bad++;
  }
  EXPECT_EQ(bad, 0);
}

TEST_P(Testbed32, DeltaBorderGeometry) {
  // Delta at the (0,0) corner --> border clipping correctness.
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout in{32, 1, 4};
  std::vector<float> w(1 * 4 * 3 * 3, 0.0f), b(1, 0.0f);
  for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
      w[(0 * 4 + 0) * 9 + di * 3 + dj] = 1.0f + di * 3 + dj;

  ConvBN<word> conv(boot_context, in, 1, 3, 1, w.data(), 4, b.data(), 1.0,
                    1.0, kLevel);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg(kNumSlots, Complex(0, 0));
  int u_in = in.UsedSlots();
  for (int rep = in.Slot(0, 0, 0); rep < kNumSlots; rep += u_in) {
    msg[rep] = Complex(1.0, 0);
  }
  Ct ct;
  EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface_->GetEvkMap());

  TensorLayout out_lay{32, 1, 1};
  std::vector<Complex> expected(kNumSlots, Complex(0, 0));
  int u_out = out_lay.UsedSlots();
  for (int di = -1; di <= 1; di++) {
    for (int dj = -1; dj <= 1; dj++) {
      int y = 0 - di, x = 0 - dj;
      if (y < 0 || y >= 32 || x < 0 || x >= 32) continue;
      double wv = w[(di + 1) * 3 + (dj + 1)];
      for (int rep = out_lay.Slot(0, y, x); rep < kNumSlots; rep += u_out) {
        expected[rep] = Complex(wv, 0);
      }
    }
  }
  std::vector<Complex> out;
  DecryptAndDecode(out, ct);

  // dump EVERY slot with significant mass (full ring) --> reveals the
  // library's actual rotation convention: slot -> (frame, y, x)
  int shown = 0;
  for (int s = 0; s < kNumSlots && shown < 40; s++) {
    double got = out[s].real(), exp = expected[s].real();
    if (std::abs(got) > 1e-3 || std::abs(exp) > 1e-3) {
      int f = s / kFrame, rem = s % kFrame;
      std::cout << "slot " << s << " (f=" << f << ",y=" << rem / kEdge
                << ",x=" << rem % kEdge << ") expected=" << exp
                << " got=" << got << std::endl;
      shown++;
    }
  }
  int bad = 0;
  for (int s = 0; s < kNumSlots; s++) {
    if (std::abs(out[s].real() - expected[s].real()) > 1e-3) bad++;
  }
  EXPECT_EQ(bad, 0);
}

static void RunDeltaVariant(Testbed32 *tb, bool use_bsgs,
                            bool suppress_bs_swap) {
  auto context = tb->context_;
  auto interface = tb->interface_.get();
  MultiLevelCiphertext<word>::StaticInit(context->param_, context->encoder_);
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context);
  ASSERT_NE(boot_context, nullptr);

  TensorLayout in{32, 1, 4};
  std::vector<float> w(1 * 4 * 3 * 3, 0.0f), b(1, 0.0f);
  for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
      w[(0 * 4 + 0) * 9 + di * 3 + dj] = 1.0f + di * 3 + dj;

  ConvBN<word> conv(boot_context, in, 1, 3, 1, w.data(), 4, b.data(), 1.0,
                    1.0, kLevel, use_bsgs, suppress_bs_swap);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  interface->PrepareRotationKey(req);

  std::vector<Complex> msg(kNumSlots, Complex(0, 0));
  int u_in = in.UsedSlots();
  for (int rep = in.Slot(0, 8, 8); rep < kNumSlots; rep += u_in) {
    msg[rep] = Complex(1.0, 0);
  }
  Ct ct;
  tb->EncodeAndEncrypt(ct, msg, kLevel);
  conv.Evaluate(ct, ct, interface->GetEvkMap());

  std::vector<Complex> out;
  tb->DecryptAndDecode(out, ct);
  int shown = 0;
  for (int s = 0; s < kNumSlots && shown < 12; s++) {
    if (std::abs(out[s].real()) > 1e-3) {
      int f = s / kFrame, rem = s % kFrame;
      std::cout << "mass slot " << s << " (f=" << f << ",y=" << rem / kEdge
                << ",x=" << rem % kEdge << ") got=" << out[s].real()
                << std::endl;
      shown++;
    }
  }
  // stamp check: out(8-di, 8-dj) == w[di+1][dj+1]
  int bad = 0;
  for (int di = -1; di <= 1; di++) {
    for (int dj = -1; dj <= 1; dj++) {
      int s = TensorLayout{32, 1, 1}.Slot(0, 8 - di, 8 - dj);
      double exp = w[(di + 1) * 3 + (dj + 1)];
      if (std::abs(out[s].real() - exp) > 1e-3) bad++;
    }
  }
  EXPECT_EQ(bad, 0) << "use_bsgs=" << use_bsgs
                    << " suppress=" << suppress_bs_swap;
}

TEST_P(Testbed32, DISABLED_DeltaSingleGroupBaby) {
  RunDeltaVariant(this, false, true);   // all rots as inner keys (avg_pool style)
}

TEST_P(Testbed32, DeltaSingleGroupSwap) {
  RunDeltaVariant(this, false, false);  // auto-swap to all-GS (proven path)
}

INSTANTIATE_TEST_SUITE_P(
    CheddarOps, Testbed32, testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
