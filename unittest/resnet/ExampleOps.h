#pragma once
// Reimplementation of the closed-source example_ops (ConvBN / DownSample /
// EvalReLU) for the open-source Cheddar release, plus Conv1x1 for MemoryBlock.
// Header-only on purpose: the library (.so) is not modified at all.
//
// Layout convention (derived from the public avg-pool code in the AE repo,
// see PORTING.md §1):  slot(c,y,x) = f*1024 + (y*k+dy)*32 + (x*k+dx)
// with f = c/(k*k), dy = (c%(k*k))/k, dx = c%k.  All ciphertexts run with
// num_slots = kNumSlots (16384); content is periodic with the stage's
// UsedSlots() and every mask is replicated accordingly.

#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/Hoist.h"

namespace cheddar {
namespace example_ops {

constexpr int kNumSlots = 1 << 14;   // block-phase slot count (= AE 1<<14)
constexpr int kFrame = 1024;         // 32x32 spatial frame
constexpr int kEdge = 32;

struct TensorLayout {
  int width;     // spatial width (= height)
  int pack;      // multiplexing factor k (width * pack == 32)
  int channels;  // number of channels C

  int FrameCount() const { return channels / (pack * pack); }
  int UsedSlots() const { return FrameCount() * kFrame; }
  int Slot(int c, int y, int x) const {
    int k = pack;
    int f = c / (k * k), sub = c % (k * k);
    return f * kFrame + (y * k + sub / k) * kEdge + (x * k + sub % k);
  }
};

// Builds the hoist map of a KxK convolution (stride 1 or 2). For stride 2 the
// output layout must be (width/2, pack*2, C_out); the rotation for a fixed
// (c_in, c_out, tap) is (y,x)-independent in both cases (PORTING.md §2).
// weights: float[C_out][C_in_w][K][K] (C_in_w = real weight input channels,
// may be smaller than layout channels, e.g. conv0 has 3 real of 4 padded).
// Every weight is multiplied by w_scale (e.g. 1/relu_range folding).
// BSGS giant-step stride: rotations decompose as (frame shift)*1024 +
// (small spatial delta), so grouping by multiples of 1024 shares giant-step
// keys across all convs and keeps baby-step keys to the few spatial deltas.
constexpr int kGsStride = 1024;

inline PlainHoistMap BuildConvHoistMap(const TensorLayout &in,
                                       const TensorLayout &out, int ksize,
                                       int stride, const float *weights,
                                       int c_in_w, double w_scale) {
  const int pad = ksize / 2;
  const int u_in = in.UsedSlots();
  const int u_out = out.UsedSlots();
  std::map<int, Message> group;  // total rotation -> mask (BSGS split below)

  for (int co = 0; co < out.channels; co++) {
    for (int ci = 0; ci < c_in_w; ci++) {
      for (int di = -pad; di <= pad; di++) {
        for (int dj = -pad; dj <= pad; dj++) {
          double wv =
              static_cast<double>(
                  weights[((static_cast<size_t>(co) * c_in_w + ci) * ksize +
                           (di + pad)) *
                              ksize +
                          (dj + pad)]) *
              w_scale;
          if (wv == 0.0) continue;
          // rotation amount: constant over (y,x); use any in-bounds point.
          int y0 = pad, x0 = pad;  // guaranteed valid for width > ksize
          int rot = in.Slot(ci, stride * y0 + di, stride * x0 + dj) -
                    out.Slot(co, y0, x0);
          rot %= u_in;
          if (rot < 0) rot += u_in;
          auto it = group.find(rot);
          if (it == group.end()) {
            it = group.try_emplace(rot, Message(kNumSlots, Complex(0, 0)))
                     .first;
          }
          Message &msg = it->second;
          for (int y = 0; y < out.width; y++) {
            int yy = stride * y + di;
            if (yy < 0 || yy >= in.width) continue;
            for (int x = 0; x < out.width; x++) {
              int xx = stride * x + dj;
              if (xx < 0 || xx >= in.width) continue;
              int s = out.Slot(co, y, x);
              for (int rep = s; rep < kNumSlots; rep += u_out) {
                msg[rep] += Complex(wv, 0);
              }
            }
          }
        }
      }
    }
  }
  // BSGS split: rot = gs + bs with gs a multiple of kGsStride; the inner
  // mask is pre-rotated by gs (LinearTransform convention:
  // stored[(i + gs) % num_slots] = mask[i]). All-zero masks are dropped.
  PlainHoistMap hoist_map;
  for (const auto &[rot, msg] : group) {
    bool all_zero = true;
    for (const auto &v : msg) {
      if (v != Complex(0, 0)) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) continue;
    int bs = rot % kGsStride;
    int gs = rot - bs;
    if (hoist_map.find(gs) == hoist_map.end()) {
      hoist_map.try_emplace(gs, std::map<int, Message>());
    }
    auto &pre_rotated =
        hoist_map[gs].try_emplace(bs, Message(kNumSlots, Complex(0, 0)))
            .first->second;
    for (int i = 0; i < kNumSlots; i++) {
      pre_rotated[(i + gs) % kNumSlots] = msg[i];
    }
  }
  return hoist_map;
}

// Per-channel bias replicated over the spatial grid (and slot periodicity).
inline std::vector<Complex> BuildBiasMessage(const TensorLayout &out,
                                             const float *bias,
                                             double b_scale) {
  std::vector<Complex> msg(kNumSlots, Complex(0, 0));
  const int u_out = out.UsedSlots();
  for (int c = 0; c < out.channels; c++) {
    double bv = static_cast<double>(bias[c]) * b_scale;
    for (int y = 0; y < out.width; y++) {
      for (int x = 0; x < out.width; x++) {
        int s = out.Slot(c, y, x);
        for (int rep = s; rep < kNumSlots; rep += u_out) {
          msg[rep] += Complex(bv, 0);
        }
      }
    }
  }
  return msg;
}

// Generic packed convolution + folded BN bias. Consumes exactly 1 level.
template <typename word>
class ConvBN {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

  std::shared_ptr<BootContext<word>> context_;
  TensorLayout in_, out_;
  int eval_level_;  // ciphertext level right before this conv
  std::unique_ptr<HoistHandler<word>> hoist_;
  Pt bias_;

  ConvBN(std::shared_ptr<BootContext<word>> context, const TensorLayout &in,
         int out_channels, int ksize, int stride, const float *weights,
         int c_in_w, const float *bias, double w_scale, double b_scale,
         int eval_level)
      : context_{context}, in_{in}, eval_level_{eval_level} {
    out_ = TensorLayout{in.width / stride, in.pack * stride, out_channels};
    auto hoist_map =
        BuildConvHoistMap(in_, out_, ksize, stride, weights, c_in_w, w_scale);
    hoist_ = std::make_unique<HoistHandler<word>>(
        context, hoist_map, eval_level,
        context->param_.GetScale(eval_level), true);
    auto bias_msg = BuildBiasMessage(out_, bias, b_scale);
    context->encoder_.Encode(bias_, eval_level - 1,
                             context->param_.GetScale(eval_level - 1),
                             bias_msg);
  }

  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    hoist_->Evaluate(context_, res, ct, evk_map);
    context_->Add(res, res, bias_);
  }

  void AddRequiredRotations(EvkRequest &req) {
    hoist_->AddRequiredRotations(req);
  }
};

// Baseline shortcut projection: 1x1 stride-2 conv (+ folded BN).
template <typename word>
class DownSample : public ConvBN<word> {
 public:
  DownSample(std::shared_ptr<BootContext<word>> context,
             const TensorLayout &in, int out_channels, const float *weights,
             const float *bias, double b_scale, int eval_level)
      : ConvBN<word>{context, in,          out_channels, 1,       2,
                     weights, in.channels, bias,         1.0,     b_scale,
                     eval_level} {}
};

// MemoryBlock building block: 1x1 stride-1 conv (channel mixing only).
template <typename word>
class Conv1x1 : public ConvBN<word> {
 public:
  Conv1x1(std::shared_ptr<BootContext<word>> context, const TensorLayout &in,
          int out_channels, const float *weights, const float *bias,
          int eval_level)
      : ConvBN<word>{context, in,     out_channels, 1,   1,
                     weights, in.channels, bias,    1.0, 1.0,
                     eval_level} {}
};

// ct-ct multiply idiom of this library (see src/extension/EvalPoly.cpp):
// dyadic MultUnsafe at (level+1) followed by RelinearizeRescale.
template <typename word>
inline void CtMult(ConstContextPtr<word> context, Ciphertext<word> &res,
                   const Ciphertext<word> &a, const Ciphertext<word> &b,
                   const EvaluationKey<word> &mult_key, int result_level) {
  Ciphertext<word> tmp;
  context->MultUnsafe(tmp, a, b, result_level + 1);
  context->RelinearizeRescale(res, tmp, mult_key);
}

// Boots (if below) or level-downs (if above) the ciphertext to `level`.
// Mirrors AdjustLevelWithBoot from the AE ResNet workload.
template <typename word>
inline void AdjustLevel(std::shared_ptr<BootContext<word>> context,
                        Ciphertext<word> &ct, int level,
                        const EvkMap<word> &evk_map) {
  if (context->param_.NPToLevel(ct.GetNP()) < level) {
    context->Boot(ct, ct, evk_map, false);
  }
  if (context->param_.NPToLevel(ct.GetNP()) > level) {
    context->LevelDown(ct, ct, level);
  }
}

// ReLU via composite minimax sign polynomials (coefficients in SignCoeffs.h):
//   s = p3(p2(p1(x))),  relu(x) = 0.5*x + 0.5*x*s   on the domain [-1, 1].
// The level schedule is fixed at construction; bootstrapping is inserted
// between stages whenever the remaining budget cannot fit the next stage.
template <typename word>
class EvalReLU {
 public:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

  std::shared_ptr<BootContext<word>> context_;
  std::vector<std::unique_ptr<EvalPoly<word>>> stages_;
  std::vector<int> stage_in_levels_;
  int boot_end_level_;
  int final_level_;    // level of s and x right before the final multiply
  int output_level_;   // final_level_ - 1

  static int DepthOf(int degree) {
    int depth = 0;
    while ((1 << depth) < degree + 1) depth++;
    return depth;
  }

  // stage_coeffs: normal-basis coefficient vectors (low degree first).
  EvalReLU(std::shared_ptr<BootContext<word>> context, int start_level,
           int boot_end_level,
           const std::vector<std::vector<double>> &stage_coeffs)
      : context_{context}, boot_end_level_{boot_end_level} {
    int lvl = start_level;
    for (const auto &coeffs : stage_coeffs) {
      int depth = DepthOf(static_cast<int>(coeffs.size()) - 1);
      if (lvl - depth < 1) lvl = boot_end_level;  // boot before this stage
      stage_in_levels_.push_back(lvl);
      auto poly = std::make_unique<EvalPoly<word>>(
          coeffs, lvl, context->param_.GetScale(lvl),
          context->param_.GetScale(lvl - depth), false);
      poly->Compile(context);
      lvl -= depth;
      stages_.push_back(std::move(poly));
    }
    if (lvl < 2) lvl = boot_end_level;  // ensure the final multiply fits
    final_level_ = lvl;
    output_level_ = lvl - 1;
  }

  int OutputLevel() const { return output_level_; }

  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    const Evk &mult_key = evk_map.GetMultiplicationKey();
    Ct s;
    context_->Copy(s, ct);
    for (size_t i = 0; i < stages_.size(); i++) {
      AdjustLevel(context_, s, stage_in_levels_[i], evk_map);
      stages_[i]->Evaluate(context_, s, s, mult_key);
    }
    AdjustLevel(context_, s, final_level_, evk_map);
    // q = 0.5*s + 0.5 (constant ops, no extra ct-ct depth)
    Constant<word> half;
    context_->encoder_.EncodeConstant(half, final_level_,
                                      context_->param_.GetScale(final_level_),
                                      0.5, 0);
    Ct q_raw, q;
    context_->Mult(q_raw, s, half);
    context_->Rescale(q, q_raw);
    Constant<word> half_add;
    context_->encoder_.EncodeConstant(
        half_add, context_->param_.NPToLevel(q.GetNP()),
        context_->param_.GetScale(context_->param_.NPToLevel(q.GetNP())), 0.5,
        0);
    context_->Add(q, q, half_add);
    Ct x;
    context_->Copy(x, ct);
    int q_level = context_->param_.NPToLevel(q.GetNP());
    AdjustLevel(context_, x, q_level, evk_map);
    CtMult<word>(context_, res, x, q, mult_key, q_level - 1);
  }
};

}  // namespace example_ops
}  // namespace cheddar
