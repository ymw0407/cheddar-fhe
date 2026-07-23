#pragma once
// MemoryBlock (memOFF variant) FHE operator for the compressed ResNet-20.
// Reuses the verified example_ops (ConvBN / EvalPoly) — no library change.
// See FHE-research/scripts/export_memresnet.py for the fused weights and
// PORTING.md for the scale convention.
//
// memOFF block (memory bank dropped — the FHE deployment regime: ~= memON
// accuracy but slot-budget-friendly, and NO sign-ReLU inside the block):
//     out = shortcut(h) + rup( rP( rdown(h) ) )
//   rdown : Cin->b  (rdown_k x rdown_k, stride, folded prenorm BN, bias)
//   rP    : elementwise low-degree polynomial (degree rdegree)
//   rup   : b->Cout (1x1, no bias)
//   shortcut : Cin->Cout (1x1, stride) or identity
// No bootstrapping inside; the caller boots between blocks.
//
// Scale convention (single global S, from manifest scale_S): the ciphertext
// carries h_norm = h_true / S so boot inputs stay bounded, while conv weights
// absorb S so the non-homogeneous polynomial rP still sees a true-scale input:
//   rdown  w_scale = S   (bias stays true-scale, added to true-scale output)
//   rup    w_scale = 1/S (renormalizes the datapath back to /S)
//   shortcut w_scale = 1 (sc_norm = W*h_norm = (W*h_true)/S directly)

#include <memory>
#include <string>
#include <vector>

#include "ExampleOps.h"
#include "common/Assert.h"

namespace cheddar {
namespace example_ops {

// Single low-degree polynomial on a true-scale ciphertext (one EvalPoly stage).
template <typename word>
class PolyAct {
 public:
  using Ct = Ciphertext<word>;
  std::shared_ptr<BootContext<word>> context_;
  std::unique_ptr<EvalPoly<word>> poly_;
  int in_level_, out_level_;

  static int DepthOf(int degree) {  // matches EvalReLU::DepthOf
    int depth = 0;
    while ((1 << depth) < degree + 1) depth++;
    return depth;
  }

  PolyAct(std::shared_ptr<BootContext<word>> context,
          const std::vector<double> &coeffs, int in_level)
      : context_{context}, in_level_{in_level} {
    int depth = DepthOf(static_cast<int>(coeffs.size()) - 1);
    out_level_ = in_level - depth;
    poly_ = std::make_unique<EvalPoly<word>>(
        coeffs, in_level, context->param_.GetScale(in_level),
        context->param_.GetScale(out_level_), false);
    poly_->Compile(context);
  }

  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    poly_->Evaluate(context_, res, ct, evk_map.GetMultiplicationKey());
  }
};

// One compressed block. Handles both variants of the mentor's block
//   C(h) = shortcut(h) + alpha*V*P(K*W_q h)  +  R_psi(h)
//                        \____ memory bank ____/   \__ residual __/
// memOFF (bank omitted) passes qk_w == nullptr. Both paths consume the same
// number of levels (conv 1 + poly depth 2 + conv 1 = 4) and run in parallel,
// so enabling the bank costs extra work but NOT extra depth or bootstraps.
//
// Slot budget: the bank's intermediate holds N channels at the block's output
// resolution, so it must satisfy N/(pack^2) * 1024 <= kNumSlots. Per layer that
// is N<=16 (32x32), N<=64 (16x16), N<=256 (8x8) — see the exporter's n-ladder.
template <typename word>
class MemBlock {
 public:
  using Ct = Ciphertext<word>;

  std::shared_ptr<BootContext<word>> context_;
  TensorLayout in_, out_;
  int in_level_, out_level_;
  bool has_memory_ = false;
  std::unique_ptr<ConvBN<word>> qk_;   // fused W_q then K (+ prenorm BN)
  std::unique_ptr<PolyAct<word>> p_;   // P (softmax surrogate)
  std::unique_ptr<ConvBN<word>> v_;    // V (alpha folded in)
  std::unique_ptr<ConvBN<word>> rdown_;
  std::unique_ptr<PolyAct<word>> rp_;
  std::unique_ptr<ConvBN<word>> rup_;
  std::unique_ptr<ConvBN<word>> shortcut_;  // null => identity

  // rdown_w [b][Cin][k][k], rdown_b [b], rP_coef [rdeg+1] (low->high),
  // rup_w [Cout][b][1][1], sc_w [Cout][Cin][1][1] (or nullptr for identity).
  // Memory bank (optional): qk_w [N][Cin][1][1], qk_b [N], P_coef, v_w [Cout][N][1][1].
  MemBlock(std::shared_ptr<BootContext<word>> context,
           const TensorLayout &in, int out_channels, int stride,
           int rdown_k, int b_channels, const float *rdown_w,
           const float *rdown_b, const std::vector<double> &rP_coef,
           const float *rup_w, const float *sc_w, double S, int in_level,
           int n_bank = 0, const float *qk_w = nullptr,
           const float *qk_b = nullptr,
           const std::vector<double> &P_coef = {},
           const float *v_w = nullptr)
      : context_{context}, in_{in}, in_level_{in_level} {
    out_ = TensorLayout{in.width / stride, in.pack * stride, out_channels};
    // rdown: Cin -> b (folds S into weights, bias true-scale). -> in_level-1.
    rdown_ = std::make_unique<ConvBN<word>>(
        context, in, b_channels, rdown_k, stride, rdown_w, in.channels,
        rdown_b, /*w_scale=*/S, /*b_scale=*/1.0, in_level);
    rp_ = std::make_unique<PolyAct<word>>(context, rP_coef, in_level - 1);
    // rup: b -> Cout (1x1) at rP's output level, w_scale = 1/S. -> out_level.
    std::vector<float> rup_bias(out_channels, 0.0f);
    rup_ = std::make_unique<ConvBN<word>>(
        context, rdown_->out_, out_channels, 1, 1, rup_w, b_channels,
        rup_bias.data(), /*w_scale=*/1.0 / S, /*b_scale=*/1.0,
        rp_->out_level_);
    out_level_ = rp_->out_level_ - 1;

    // ---- memory bank (mentor's alpha*V*P(K*W_q h)) --------------------------
    if (qk_w != nullptr) {
      has_memory_ = true;
      TensorLayout bank{out_.width, out_.pack, n_bank};
      AssertTrue(bank.UsedSlots() <= kNumSlots,
                 "memory bank does not fit one ciphertext: N=" +
                     std::to_string(n_bank) + " at width " +
                     std::to_string(out_.width) + " needs " +
                     std::to_string(bank.UsedSlots()) + " slots (limit " +
                     std::to_string(kNumSlots) + ") — lower N for this layer");
      // qk: Cin -> N (1x1, folds S and the prenorm BN). -> in_level-1.
      qk_ = std::make_unique<ConvBN<word>>(
          context, in, n_bank, 1, stride, qk_w, in.channels, qk_b,
          /*w_scale=*/S, /*b_scale=*/1.0, in_level);
      p_ = std::make_unique<PolyAct<word>>(context, P_coef, in_level - 1);
      AssertTrue(p_->out_level_ == rp_->out_level_,
                 "memory and residual paths must land on the same level");
      // v: N -> Cout (1x1), alpha already folded by the exporter, 1/S renorm.
      std::vector<float> v_bias(out_channels, 0.0f);
      v_ = std::make_unique<ConvBN<word>>(
          context, qk_->out_, out_channels, 1, 1, v_w, n_bank,
          v_bias.data(), /*w_scale=*/1.0 / S, /*b_scale=*/1.0, p_->out_level_);
    }

    if (sc_w != nullptr) {  // shortcut maps h_norm -> sc_norm directly (w=1).
      std::vector<float> sc_bias(out_channels, 0.0f);
      shortcut_ = std::make_unique<ConvBN<word>>(
          context, in, out_channels, 1, stride, sc_w, in.channels,
          sc_bias.data(), /*w_scale=*/1.0, /*b_scale=*/1.0, in_level);
    }
  }

  const TensorLayout &OutLayout() const { return out_; }
  int InLevel() const { return in_level_; }
  int OutLevel() const { return out_level_; }

  bool HasMemory() const { return has_memory_; }

  void AddRequiredRotations(EvkRequest &req) {
    rdown_->AddRequiredRotations(req);
    rup_->AddRequiredRotations(req);
    if (qk_) qk_->AddRequiredRotations(req);
    if (v_) v_->AddRequiredRotations(req);
    if (shortcut_) shortcut_->AddRequiredRotations(req);
  }

  // Input ct must sit at in_level_ (caller boots/levels beforehand).
  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    Ct r;
    rdown_->Evaluate(r, ct, evk_map);   // in_level-1, true scale
    rp_->Evaluate(r, r, evk_map);       // rP out level
    rup_->Evaluate(r, r, evk_map);      // out_level, normalized
    if (has_memory_) {                  // parallel path, same out level
      Ct m;
      qk_->Evaluate(m, ct, evk_map);
      p_->Evaluate(m, m, evk_map);
      v_->Evaluate(m, m, evk_map);
      context_->Add(r, r, m);
    }
    Ct sc;
    if (shortcut_) {
      shortcut_->Evaluate(sc, ct, evk_map);  // in_level-1
    } else {
      context_->Copy(sc, ct);                // identity
    }
    context_->LevelDown(sc, sc, out_level_);
    context_->Add(res, r, sc);
  }
};

}  // namespace example_ops
}  // namespace cheddar
