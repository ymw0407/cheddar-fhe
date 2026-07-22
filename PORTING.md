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

### Conv (stride 1, K×K, 레이아웃 유지)
(c_in, tap di,dj) → c_out 의 회전량은 (y,x) 무관 상수:
r = (f_in−f_out)·1024 + (dy_in−dy_out+di·k)·32 + (dx_in−dx_out+dj·k), mod u.
마스크는 전 슬롯 브루트포스: 출력슬롯 s_out 에 대해 (y+di, x+dj) 가 경계 안일 때만 weight 값
(padding=1 경계는 마스크 0). → HoistHandler 1개, 레벨 1 소비.

### Narrowing Conv (stride 2, C→2C, k→2k) — 2단계 (AE 의 conv_weight_+conv_mask_ 구조와 동일)
- 검증된 핵심: slot_old(c, 2y, 2x) 는 (y,x) 에 대해 새 레이아웃과 같은 선형계수 → 단계B 회전도 상수.
- 단계 A: stride-1 형태의 conv hoist (출력을 입력 정렬 위치에, c_out 을 옛 레이아웃 좌표에 배치).
- 단계 B: c_out 별 상수 회전 + 짝수 (y,x) 만 선택하는 마스크 → 새 레이아웃 (w/2, 2k).
- 레벨 2 소비 (= AE conv_level=2 정합).

### DownSample (baseline 숏컷) = 1×1 tap 의 narrowing (단계 A tap 1개 + 단계 B).
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
- [x] ExampleOps.h 초판 / ResNet.cpp 포트 / CMake / 계수 생성기
- [ ] 서버 빌드 1라운드 (컴파일 에러 수정 반복 예상 — API 드리프트)
- [ ] 평문 대조 (CompareMessages) → conv 마스크 검증
- [ ] baseline 시간·정확도 (anchor: 공표 1.32s/A100, acc ~91.3%)
- [ ] MemResNet.cpp + export (FHE-research 쪽 P2·P3 과 합류)

## 5. 검증 포인트 (서버에서 확인 필요)
1. Encode 시 msg 길이 = num_slots 의미론 (16384 명시 인코딩으로 회피)
2. HoistHandler gs 그룹핑: v1 은 gs=0 단일 그룹 (키 수 많음 — 동작 우선, BSGS 최적화는 후속)
3. EvalPoly target_scale 관례 (param_->GetScale(level))
4. Boot 후 레벨 (boot_param_.GetEndLevel()) 과 AdjustLevelWithBoot 흐름
