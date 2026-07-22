# MemoryBlock 포팅 작업 문서 (FHE-research → Cheddar)

목표: 같은 라이브러리·같은 A100에서 **baseline ResNet-20 vs MemoryBlock 압축 ResNet-20** 실측.
전략: 라이브러리(.so 소스)는 건드리지 않고 **워크로드 계층(unittest/resnet/)에 헤더-온리로** 전부 구현.

## 1. 레이아웃 규약 (공개된 avg_pool 코드에서 역산 — 증명됨)

- 슬롯 공간: half_degree = 2^15. 블록 구간 운용 num_slots = **16384** (= AE의 PrepareBoot(1<<14)).
- 프레임 = 32×32 spatial (1024슬롯). 16384 = 16프레임.
- 스테이지 (width w, pack k, channels C): 채널 c → frame f = c/(k·k), sub = c%(k·k), dy = sub/k, dx = sub%k.
  **slot(c,y,x) = f·1024 + (y·k+dy)·32 + (x·k+dx)**, 사용 슬롯 u = (C/k²)·1024.
- u < 16384 이면 내용이 주기 u 로 **복제**되어야 함 (회전 wrap 정합). 모든 마스크는 [0,u) 에서 만들고
  16384 까지 주기 복제. 회전량은 mod u 로 정규화.
- 검증: 이 규약으로 ResNet.cpp의 avg_pool 마스크(rot_src = i·1024+j·32 → rot_dst = 4(i·4+j))와
  Trace 시퀀스가 정확히 재현됨 → pool 후 채널 c 가 슬롯 c 에 위치 → FC 로 연결.

## 2. 연산자 설계

### Conv (stride 1·2 공통, K×K) — 단일 단계, 레벨 1 소비
(c_in, tap di,dj) → c_out 의 회전량은 (y,x) 무관 상수 (stride 2 포함 — 증명됨):
r = in.Slot(ci, s·y+di, s·x+dj) − out.Slot(co, y, x), mod u_in.
마스크는 전 슬롯 브루트포스, padding 경계는 0. **복제 보정**: u_out < u_in 인
narrowing 단계에서 출력 복제본 k(+k·u_out 위치)는 회전 (r − k·u_out) mod u_in 을
써야 함 (동일 회전으로 복제하면 복제 영역이 타 채널 쓰레기 — round6 에서 실증).

### 평가 = 수동 BSGS (HoistHandler 미사용)
- r = gs + bs (bs = r % 32). 입력을 baby 회전(HRot, ≤14개)해 두고, giant 그룹별로
  (gs 만큼 사전회전한 마스크)·(baby 회전 입력) 을 Mult(ct×pt)로 누적 → HRotAdd(gs).
  마지막에 Rescale 1회 + bias(레벨 ℓ−1) Add. 키 = |baby|+|giant| ≈ 층당 ~50개.
- 라이브러리 HoistHandler 의 inner-key(baby) 경로는 우리 맵에서 bs≠0 기여를
  통째로 소실시킴 (델타 프로브로 실증; FC/부트의 LinearTransform 맵은 정상).
  auto-swap all-GS 경로는 정확하지만 회전키가 회전 수만큼(L3 ~500개) 필요해
  키 메모리 OOM → 수동 BSGS 가 정확성·키 수·키스위치 수 모두 해결.

### DownSample (baseline 숏컷) = 1×1 stride-2 conv (위 규약 그대로, b_scale 폴드).
### Conv1x1 (우리 MemBlock 용) = 1-tap stride-1 conv → 프레임 간 회전만, 레벨 1.

### 스케일 규약
- 전 활성은 relu_range(=10) 로 정규화되어 흐름: conv0 와 모든 bias 에 1/10 폴드,
  이후 conv 는 정규화 유지(추가 스케일 없음), pool 마스크가 ×10 복원 (AE 코드 그대로).
- EvalReLU 는 [-1,1] 정규화 도메인에서 동작. relu(x/10)=relu(x)/10 이라 정합.

### EvalReLU (baseline 용 재구현)
- relu(x) = x·(1+sgn(x))/2, sgn = p3∘p2∘p1 (홀수 합성 다항식, minimax).
- 각 단계 = EvalPoly (Chebyshev basis 권장), 마지막 = 0.5·x + 0.5·x·s.
- 계수는 unittest/resnet/gen_sign_coeffs.py 로 생성 → SignCoeffs.h (교체 가능).
- 우리 MemBlock 은 EvalReLU 불필요 (P deg3·σ deg2 = EvalPoly 직접).

## 3. 파일 구성
- unittest/resnet/ExampleOps.h — 위 전부 (헤더-온리, 라이브러리 무수정)
- unittest/resnet/SignCoeffs.h — sgn 합성 계수 (생성물)
- unittest/resnet/gen_sign_coeffs.py — 계수 생성기 (Remez/minimax)
- unittest/resnet/ResNet.cpp — baseline 포트 (AE 템플릿, cnpy·Eigen FetchContent)
- unittest/resnet/DatasetUtils.h — AE 복사 (CIFAR-10 로더)
- unittest/resnet/MemResNet.cpp — (P4) 우리 압축 모델
- parameters/resnetparam_*.json — AE 복사
- resnet20_fused/ — AE 가중치 복사 (npy)

## 4. 상태
- [x] 정찰·레이아웃 증명·설계 (2026-07-22)
- [x] ExampleOps.h / ResNet.cpp 포트 / CMake / 계수 생성기
- [x] 서버 빌드·E2E 실행 (A100, cuda-12.6, sm_80, USE_GMP)
- [x] 버그 수정 4연타: bias 1/10 폴드 누락 → 키 재생성 스킵(레벨 오염) →
      HoistHandler baby 경로 bs≠0 소실(수동 BSGS 로 우회) → narrowing 복제 보정
- [x] ops_test 기하 8종 통과 (delta stamp·경계·conv0 실가중치·CIFAR 크기)
- [x] round6: conv0 max 평문 완전일치(0.138545), L1 3블록 궤적 일치, L2 생존
- [x] round7: 수동 BSGS 로 L3 관문 통과, E2E 3.40s — 남은 문제 = FC 로짓 폭주
- [x] round8: FC 도 ManualLinear(수동 BSGS 대각법) 로 교체 → **로짓 정합,
      pred=3 정답, Accuracy 1 (3.41s) = P1 정확성 완료** (2026-07-22)
      로짓 편차 ±0.3~0.9 는 sign 합성(전이 0.05) 근사 노이즈 — argmax 마진 충분
- [ ] IMAGES=n 다중 이미지 FHE 정확도 (목표: 평문 fused 모델과 동률 근처)
- [ ] baseline 시간·정확도 (anchor: 공표 1.32s/A100, acc ~91.3%)
- [ ] MemResNet.cpp + export (FHE-research 쪽 P2·P3 과 합류)

## 5. 운영 메모 (서버)
1. 회전키는 층 그룹별 load/drop (EraseRotationKey) — 존재해도 절대 스킵 금지
   (레벨 부족 키 재사용 = 쓰레기 회전; PrepareRotationKey 가 내부에서 알아서 재생성).
2. "WARN: already prepared" 는 공유 키에 대한 무해한 잡음.
3. EvalPoly target_scale 관례 (param_->GetScale(level)) — SignCoeffs 3단 합성 검증됨.
4. Boot 후 레벨 = boot_end_level(8); EvalReLU 가 내부에서 단계별 AdjustLevel.
