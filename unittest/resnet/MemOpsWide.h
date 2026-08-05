#pragma once
// Multi-ciphertext (wide-b) MemoryBlock — b512/b1024 잔차 폭 회로 (2026-08-05).
//
// 새 회로 실험은 공유 파일(MemOps.h / MemResNet.cpp)을 건드리지 않는다는
// FHE-research 규약에 따라 별도 파일. 배경: 병목 진단(용량)의 해결책 ⓐ —
// 잔차 폭 b 를 스테이지 슬롯 캡(16/64/256) 너머로 키우면 평문 +2.0pp(b512
// 0.6356) / +3.8pp(b1024 0.6537) 실측. 이 파일은 그 학생을 FHE 로 올린다.
//
// 구조: b_total 을 ceil(b/cap) 개의 암호문 조각으로 쪼갠다.
//   rdown_p : Cin -> b_p   (출력 채널 슬라이스 — 각 조각이 별도 ct)
//   rP      : 조각별 동일 다항식 (elementwise, 같은 계수)
//   rup_p   : b_p -> Cout  (입력 채널 슬라이스 — 결과를 합산)
//   out = shortcut(h) + sum_p rup_p( rP( rdown_p(h) ) )
// 수학적으로 rup(rP(rdown(h))) 와 동일 (rP 가 원소별이라 채널 분할과 교환,
// rup 는 입력 채널에 선형이라 조각 합 = 전체). 시간은 조각 수에 비례
// (예측 0.109s x ct합 — 이 실측이 본 실험의 목적).
//
// 메모리 뱅크는 미지원 — b512/b1024 학생은 memOFF (driver 가 거부).

#include <algorithm>
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
class MemBlockWide {
 public:
  using Ct = Ciphertext<word>;

  std::shared_ptr<BootContext<word>> context_;
  TensorLayout in_, out_;
  int in_level_, out_level_;
  int parts_ = 1;
  std::vector<std::unique_ptr<ConvBN<word>>> rdown_;
  std::vector<std::unique_ptr<PolyAct<word>>> rp_;
  std::vector<std::unique_ptr<ConvBN<word>>> rup_;
  std::unique_ptr<ConvBN<word>> shortcut_;  // null => identity

  // rdown_w [b][Cin][k][k] (출력 채널 = 행 연속 -> 포인터 오프셋 슬라이스),
  // rdown_b [b], rup_w [Cout][b][1][1] (입력 채널 슬라이스 = 스트라이드 복사).
  // 스케일 규약은 MemBlock 과 동일: rdown w*=S(bias 실척도), rup w*=1/S.
  MemBlockWide(std::shared_ptr<BootContext<word>> context,
               const TensorLayout &in, int out_channels, int stride,
               int rdown_k, int b_total, const float *rdown_w,
               const float *rdown_b, const std::vector<double> &rP_coef,
               const float *rup_w, const float *sc_w, double S, int in_level)
      : context_{context}, in_{in}, in_level_{in_level} {
    out_ = TensorLayout{in.width / stride, in.pack * stride, out_channels};
    const int pack_out = in.pack * stride;
    // 한 암호문이 담는 채널 캡: FrameCount*kFrame <= kNumSlots
    //   -> cap = (kNumSlots/kFrame) * pack_out^2  (스테이지별 16/64/256)
    const int cap = (kNumSlots / kFrame) * pack_out * pack_out;
    AssertTrue(b_total > 0, "MemBlockWide: b_total must be positive");
    parts_ = (b_total + cap - 1) / cap;
    std::cout << " [wide b=" << b_total << " cap=" << cap
              << " parts=" << parts_ << "]" << std::flush;
    const int cin = in.channels;
    for (int p = 0; p < parts_; p++) {
      const int b0 = p * cap;
      const int bp = std::min(cap, b_total - b0);
      rdown_.push_back(std::make_unique<ConvBN<word>>(
          context, in, bp, rdown_k, stride,
          rdown_w + static_cast<size_t>(b0) * cin * rdown_k * rdown_k, cin,
          rdown_b + b0, /*w_scale=*/S, /*b_scale=*/1.0, in_level));
      rp_.push_back(
          std::make_unique<PolyAct<word>>(context, rP_coef, in_level - 1));
      // rup 조각: 행마다 [b0, b0+bp) 열을 스트라이드 b_total 에서 복사.
      // ConvBN 이 ctor 에서 마스크로 복사하므로 지역 버퍼면 충분하다.
      std::vector<float> rup_slice(static_cast<size_t>(out_channels) * bp);
      for (int co = 0; co < out_channels; co++)
        for (int j = 0; j < bp; j++)
          rup_slice[static_cast<size_t>(co) * bp + j] =
              rup_w[static_cast<size_t>(co) * b_total + b0 + j];
      std::vector<float> rup_bias(out_channels, 0.0f);
      rup_.push_back(std::make_unique<ConvBN<word>>(
          context, rdown_.back()->out_, out_channels, 1, 1, rup_slice.data(),
          bp, rup_bias.data(), /*w_scale=*/1.0 / S, /*b_scale=*/1.0,
          rp_.back()->out_level_));
    }
    out_level_ = rp_[0]->out_level_ - 1;
    if (sc_w != nullptr) {
      std::vector<float> sc_bias(out_channels, 0.0f);
      shortcut_ = std::make_unique<ConvBN<word>>(
          context, in, out_channels, 1, stride, sc_w, in.channels,
          sc_bias.data(), /*w_scale=*/1.0, /*b_scale=*/1.0, in_level);
    }
  }

  const TensorLayout &OutLayout() const { return out_; }
  int InLevel() const { return in_level_; }
  int OutLevel() const { return out_level_; }

  void AddRequiredRotations(EvkRequest &req) {
    for (auto &c : rdown_) c->AddRequiredRotations(req);
    for (auto &c : rup_) c->AddRequiredRotations(req);
    if (shortcut_) shortcut_->AddRequiredRotations(req);
  }

  // Input ct must sit at in_level_ (caller boots/levels beforehand).
  void Evaluate(Ct &res, const Ct &ct, const EvkMap<word> &evk_map) {
    Ct r, part, t;
    for (int p = 0; p < parts_; p++) {
      rdown_[p]->Evaluate(t, ct, evk_map);
      rp_[p]->Evaluate(t, t, evk_map);
      rup_[p]->Evaluate(part, t, evk_map);
      if (p == 0) {
        context_->Copy(r, part);
      } else {
        context_->Add(r, r, part);
      }
    }
    Ct sc;
    if (shortcut_) {
      shortcut_->Evaluate(sc, ct, evk_map);
    } else {
      context_->Copy(sc, ct);
    }
    context_->LevelDown(sc, sc, out_level_);
    context_->Add(res, r, sc);
  }
};

}  // namespace example_ops
}  // namespace cheddar
