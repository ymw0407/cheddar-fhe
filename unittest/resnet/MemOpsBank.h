#pragma once
// Wide-bank MemoryBlock — 뱅크 N 을 1-암호문 슬롯 캡 너머로 (2026-08-09).
//
// 새 회로 실험은 공유 파일(MemOps.h / MemResNet.cpp)을 건드리지 않는다는
// FHE-research 규약에 따라 별도 파일. 배경: 뱅크 회생 판정(배치 조건부) —
// 천장맥스 rn32c100_lw3_free_Nceil(N=64/256/1024, 스테이지 캡 16/64/256 의
// 4배)이 평문 0.6315~0.6345 (3시드 mean 0.6330). 이 파일은 그 학생을 FHE 로
// 올린다 (사다리 배치 d5e400 의 FHE 실측 0.6318 과 같은 자로 비교).
//
// 구조: N_total 을 ceil(N/cap) 개의 암호문 조각으로 쪼갠다 (MemOpsWide.h 의
// 잔차 폭 분할과 동일한 수학을 뱅크 경로에 적용).
//   qk_g : Cin -> N_g  (qk_w 출력 행 슬라이스 — 각 조각이 별도 ct)
//   P    : 조각별 동일 다항식 (원소별 — 인스턴스 1개 공유)
//   v_g  : N_g -> Cout (v_w 입력 열 슬라이스 — 결과를 합산)
//   bank(h) = sum_g v_g(P(qk_g(h))) == v(P(qk(h)))
// (P 가 원소별이라 채널 분할과 교환하고, v 는 입력 채널에 선형이라 조각 합 =
// 전체.) 잔차 경로·shortcut·스케일 규약은 MemBlock 과 동일; 깊이도 동일
// (조각은 일을 늘릴 뿐 레벨을 늘리지 않는다). 시간은 조각 수에 비례 —
// 천장맥스는 세 스테이지 모두 4조각이라 뱅크 비용 ~4x 가 예측치.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ExampleOps.h"
#include "MemOps.h"  // PolyAct 재사용
#include "common/Assert.h"

namespace cheddar {
namespace example_ops {

template <typename word>
class MemBlockBank {
 public:
  using Ct = Ciphertext<word>;

  std::shared_ptr<BootContext<word>> context_;
  TensorLayout in_, out_;
  int in_level_, out_level_;
  bool has_memory_ = false;
  int bank_parts_ = 0;
  std::vector<std::unique_ptr<ConvBN<word>>> qk_;  // 조각별 W_q·K (+prenorm BN)
  std::unique_ptr<PolyAct<word>> p_;               // P (조각 공유, 원소별)
  std::vector<std::unique_ptr<ConvBN<word>>> v_;   // 조각별 V (alpha folded)
  std::unique_ptr<ConvBN<word>> rdown_;
  std::unique_ptr<PolyAct<word>> rp_;
  std::unique_ptr<ConvBN<word>> rup_;
  std::unique_ptr<ConvBN<word>> shortcut_;  // null => identity

  // 가중치 레이아웃·스케일 규약은 MemBlock 과 동일:
  //   rdown_w [b][Cin][k][k], rdown_b [b], rP_coef, rup_w [Cout][b][1][1],
  //   sc_w [Cout][Cin][1][1] | nullptr, qk_w [N][Cin][1][1], qk_b [N],
  //   P_coef, v_w [Cout][N][1][1]. rdown/qk w*=S, rup/v w*=1/S, sc w*=1.
  MemBlockBank(std::shared_ptr<BootContext<word>> context,
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
    const int pack_out = in.pack * stride;
    // 한 암호문이 담는 채널 캡 (MemOpsWide 와 동일 산식, 스테이지별 16/64/256)
    const int cap = (kNumSlots / kFrame) * pack_out * pack_out;
    AssertTrue(b_channels <= cap,
               "MemBlockBank: residual b=" + std::to_string(b_channels) +
                   " exceeds cap " + std::to_string(cap) +
                   " — wide-b is memresnet_wide's job, not this circuit's");
    // 잔차 경로 (MemBlock 과 동일)
    rdown_ = std::make_unique<ConvBN<word>>(
        context, in, b_channels, rdown_k, stride, rdown_w, in.channels,
        rdown_b, /*w_scale=*/S, /*b_scale=*/1.0, in_level);
    rp_ = std::make_unique<PolyAct<word>>(context, rP_coef, in_level - 1);
    std::vector<float> rup_bias(out_channels, 0.0f);
    rup_ = std::make_unique<ConvBN<word>>(
        context, rdown_->out_, out_channels, 1, 1, rup_w, b_channels,
        rup_bias.data(), /*w_scale=*/1.0 / S, /*b_scale=*/1.0,
        rp_->out_level_);
    out_level_ = rp_->out_level_ - 1;

    // ---- wide 메모리 뱅크: N 을 cap 단위 조각으로 --------------------------
    if (qk_w != nullptr) {
      has_memory_ = true;
      AssertTrue(n_bank > 0, "MemBlockBank: bank requested with N<=0");
      bank_parts_ = (n_bank + cap - 1) / cap;
      std::cout << " [bank N=" << n_bank << " cap=" << cap
                << " parts=" << bank_parts_ << "]" << std::flush;
      p_ = std::make_unique<PolyAct<word>>(context, P_coef, in_level - 1);
      if (p_->out_level_ != rp_->out_level_) {
        // 경로 깊이가 다르면 (rP deg3 vs P deg5) 깊은 쪽이 출력 레벨을 정하고
        // 얕은 쪽은 add 직전에 내린다 — MemBlock 과 동일.
        std::cout << "[bank-block] path depths differ: rP out "
                  << rp_->out_level_ << " vs P out " << p_->out_level_
                  << " — aligning at add" << std::endl;
        out_level_ =
            (rp_->out_level_ < p_->out_level_ ? rp_->out_level_
                                              : p_->out_level_) - 1;
      }
      const int cin = in.channels;
      for (int g = 0; g < bank_parts_; g++) {
        const int n0 = g * cap;
        const int ng = std::min(cap, n_bank - n0);
        // qk 조각: 출력 행 [n0, n0+ng) — 1x1 이라 행 연속, 포인터 오프셋.
        qk_.push_back(std::make_unique<ConvBN<word>>(
            context, in, ng, 1, stride,
            qk_w + static_cast<size_t>(n0) * cin, cin, qk_b + n0,
            /*w_scale=*/S, /*b_scale=*/1.0, in_level));
        // v 조각: 행마다 [n0, n0+ng) 열을 스트라이드 n_bank 에서 복사.
        // ConvBN 이 ctor 에서 마스크로 복사하므로 지역 버퍼면 충분하다.
        std::vector<float> v_slice(static_cast<size_t>(out_channels) * ng);
        for (int co = 0; co < out_channels; co++)
          for (int j = 0; j < ng; j++)
            v_slice[static_cast<size_t>(co) * ng + j] =
                v_w[static_cast<size_t>(co) * n_bank + n0 + j];
        std::vector<float> v_bias(out_channels, 0.0f);
        v_.push_back(std::make_unique<ConvBN<word>>(
            context, qk_.back()->out_, out_channels, 1, 1, v_slice.data(), ng,
            v_bias.data(), /*w_scale=*/1.0 / S, /*b_scale=*/1.0,
            p_->out_level_));
      }
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
  int BankParts() const { return bank_parts_; }

  void AddRequiredRotations(EvkRequest &req) {
    rdown_->AddRequiredRotations(req);
    rup_->AddRequiredRotations(req);
    for (auto &c : qk_) c->AddRequiredRotations(req);
    for (auto &c : v_) c->AddRequiredRotations(req);
    if (shortcut_) shortcut_->AddRequiredRotations(req);
  }

  // Input ct must sit at in_level_ (caller boots/levels beforehand).
  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    Ct r;
    rdown_->Evaluate(r, ct, evk_map);   // in_level-1, true scale
    rp_->Evaluate(r, r, evk_map);       // rP out level
    rup_->Evaluate(r, r, evk_map);      // 정규화 경로 출력
    if (has_memory_) {                  // 병렬 경로, 조각 합산
      Ct m, part, t;
      for (int g = 0; g < bank_parts_; g++) {
        qk_[g]->Evaluate(t, ct, evk_map);
        p_->Evaluate(t, t, evk_map);
        v_[g]->Evaluate(part, t, evk_map);
        if (g == 0) {
          context_->Copy(m, part);
        } else {
          context_->Add(m, m, part);
        }
      }
      int rl = context_->param_.NPToLevel(r.GetNP());
      int ml = context_->param_.NPToLevel(m.GetNP());
      int tgt = rl < ml ? rl : ml;
      if (rl > tgt) context_->LevelDown(r, r, tgt);
      if (ml > tgt) context_->LevelDown(m, m, tgt);
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
