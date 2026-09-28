# 초파리 항법 회로–몸체 제약 폐루프 시뮬레이션 연구
## 다른 채팅으로 전달하기 위한 종합 인수인계 문서

작성 목적:
이 문서는 현재까지 확정된 연구 방향, 문헌 검토 결과, 모델 선택 이유, 수학적 구조, 시뮬레이션 설계, 실험군, 평가 지표, 구현 원칙, 3D 확장 계획, 주의사항, 미확정 사항을 하나의 문서로 통합하여 다른 ChatGPT 대화에서 즉시 연구를 이어갈 수 있도록 하기 위한 인수인계 문서이다.

이 문서를 읽는 다음 작업자는 전체 연구계획을 처음부터 다시 제안하거나 선택지만 나열하지 말고, 아래의 확정사항을 연구의 최신 기준으로 삼아 실제 로컬 구현 단계로 내려가야 한다. 특히 사용자는 연구를 전부 시뮬레이션으로 수행하며, 구현은 사용자의 개인 컴퓨터에서 직접 로컬로 진행할 예정이다. 다음 채팅의 역할은 사용자가 로컬에서 직접 구현할 수 있도록 코드 구조, 파일 단위 구현, 실행 절차, 검증 절차, 오류 진단을 단계별로 안내하는 것이다.

---

# 1. 연구의 출발점과 장기 목표

이 연구의 원래 출발점은 다음 질문이다.

“초파리의 신경 커넥톰과 그로부터 알려진 행동 회로를 실제 계산 가능한 회로로 구현하면, 원래 수행하던 행동 기능도 나타날 수 있는가?”

장기 목표는 단순히 초파리 행동을 모사하는 데 있지 않다. 최종적으로는 초파리에서 유래한 공통 항법·행동 회로를 하나의 계산 코어로 만들고, 서로 다른 몸체나 이동체에 연결했을 때도 그 기능이 얼마나 보존되는지 검증하는 것이다.

장기적 개념은 다음과 같다.

공통 초파리 유래 항법 코어
→ 몸체별 인터페이스/제약 플러그인
→ 서로 다른 body dynamics
→ 실제 운동 결과
→ 센서 피드백
→ 다시 공통 코어

즉, 장기적으로는 “하나의 신경 항법 코어가 여러 종류의 몸체에 재사용될 수 있는가?”를 묻는다.

이때 매우 중요한 것은 연구의 첫 단계에서는 실제 드론이나 로봇 하드웨어를 사용하지 않는다는 점이다. 모든 연구는 시뮬레이션으로 수행한다. 하드웨어 구현, 칩 구현, 실제 드론 탑재는 장기 확장 방향일 뿐, 현재 연구 범위가 아니다.

---

# 2. 현재 연구의 핵심 질문

현재 1차 연구의 중심 질문은 다음과 같이 정리한다.

“고정된 초파리 유래 steering circuit가 자신과 다른 운동 제약을 가진 body와 폐루프로 연결되었을 때, 어느 정도의 body limitation까지 항법 기능을 유지하는가?”

더 엄밀하게는 다음 세 가지를 본다.

1. 초파리 유래 steering core가 이상적인 몸체에서는 정상적으로 목표 방향을 회복하는가?
2. 동일한 core를 전혀 재학습하거나 재조정하지 않고 body dynamics만 느리게 하거나 제한했을 때 어느 지점부터 성능이 무너지는가?
3. body-side plugin이 neural core의 목표 판단 기능을 대신하지 않으면서, 단순한 actuator mismatch를 어느 정도 완화할 수 있는가?

연구의 핵심 novelty 후보는 단순히 “초파리 회로를 시뮬레이션했다”가 아니다.

보다 중요한 프레이밍은 다음과 같다.

fixed biological / connectome-informed steering circuit
+ systematically varied embodiment
+ actual-motion feedback
+ body-constraint tolerance/failure boundary

즉, 내부 회로를 고정한 상태에서 외부 body의 속도, 가속도, 응답지연 등을 체계적으로 변화시키고, 그 결과 생기는 안정/실패 영역을 지도화하는 것이다.

---

# 3. 반드시 유지해야 할 핵심 연구 원칙

## 3.1 전부 시뮬레이션

이번 연구는 실제 하드웨어를 사용하지 않는다.

초기 단계:
- Python 기반 reduced neural circuit simulation
- abstract body dynamics
- heading recovery task

확장 단계:
- 2D 또는 3D body
- 3D 시각화
- 필요 시 MuJoCo 등의 physics engine

그러나 3D 그래픽을 사용한다고 해서 처음부터 복잡한 실제 드론 physics가 반드시 필요한 것은 아니다.

---

## 3.2 “커넥톰을 그대로 실행”한다고 표현하지 않는다

현재 사용할 첫 모델은 whole-brain raw connectome simulation이 아니다.

정확한 표현은 다음과 같다.

- Drosophila-derived steering circuit
- connectome-informed steering circuit
- reduced central-complex steering model

반대로 다음 표현은 피한다.

- “초파리 뇌 전체를 그대로 실행했다”
- “초파리 커넥톰 전체가 드론을 조종한다”
- “whole-brain fly simulation”이라는 과장된 표현

왜냐하면 첫 모델은 특정 central-complex steering pathway를 축약한 모델이기 때문이다.

---

## 3.3 core 내부는 body 조건에 따라 재학습하지 않는다

body constraint를 비교하는 본실험에서는 같은 neural core parameter를 모든 body 조건에 그대로 사용해야 한다.

예:
- 동일한 connection weights
- 동일한 activation
- 동일한 decoder
- 동일한 neural update rate

body condition마다 neural core를 다시 tune하면 “body에 대한 회로의 tolerance”를 볼 수 없게 된다.

Per-body retraining은 필요하다면 별도 comparison experiment로만 수행한다.

---

## 3.4 실제 운동 결과를 반드시 feedback한다

폐루프에서 다음은 절대 금지한다.

neural command → 다음 timestep의 heading으로 바로 사용

올바른 구조는 다음과 같다.

neural command
→ plugin
→ body dynamics
→ actual motion
→ sensor
→ neural core

즉 brain이 원하는 움직임과 실제 움직임을 반드시 분리한다.

---

## 3.5 plugin은 navigation을 대신하면 안 된다

body plugin의 역할은 “어디로 가야 하는가?”를 결정하는 것이 아니다.

plugin이 알아도 되는 정보:
- r_brain
- actual yaw rate r
- r_max
- alpha_max
- tau_r
- 기타 actuator/body capability

plugin이 알아서는 안 되는 정보:
- goal heading psi_g
- heading error psi_g - psi
- target x, y, z
- path
- obstacle map
- global navigation state

즉 plugin은 goal-blind actuator/interface layer여야 한다.

---

# 4. 선행연구 검토 결과와 모델 선택

현재까지 검토한 주요 연구들은 다음과 같다.

## 4.1 Westeinde et al., Nature 2024

논문:
“Transforming a head direction signal into a goal-oriented steering command”

현재 첫 core로 가장 적합한 후보이다.

이 연구는 초파리 central complex에서 head-direction signal과 goal signal이 PFL 계열 뉴런을 통해 steering command로 변환되는 과정을 모델링한다.

중요한 점:
- heading representation에서 끝나지 않는다.
- steering output까지 내려간다.
- PFL2, PFL3, DNa03, DNa02 pathway를 포함한다.
- closed-loop simulation을 제시한다.
- 10 Hz 업데이트를 사용한다.
- 현재 연구의 “heading recovery” task와 직접 연결 가능하다.

따라서 첫 연구의 baseline core로 선택한다.

단, 이 모델은 raw hemibrain synapse network 전체를 직접 실행하는 것이 아니라 hemibrain connectivity statistics와 physiology를 바탕으로 만든 reduced connectome-informed model이다.

---

## 4.2 Hulse et al., eLife

초파리 central complex connectome을 제공하는 중요한 해부학적 기반 자료이다.

이 연구는 회로 구조와 network motif를 이해하는 데 매우 중요하지만, 그 자체가 완전한 executable navigation controller는 아니다.

따라서 구조적 근거로 사용하되 첫 실행 core로 직접 사용하지 않는다.

---

## 4.3 Duan et al., NeurIPS 2025

“From Synapses to Dynamics...” 계열 연구.

초파리 head-direction circuit를 connectome-constrained dynamics로 구성하고, unknown cell-type gain, threshold, time constant 등을 학습하여 continuous attractor dynamics를 복원한다.

장점:
- raw connectome dynamics에 더 가깝다.
- head-direction attractor 구현에 강하다.

단점:
- 기본적으로 heading representation까지가 핵심이다.
- steering/motor output이 직접 제공되지 않는다.
- 첫 연구에 사용하면 artificial controller를 추가해야 하므로 attribution이 흐려진다.

따라서 나중의 “더 생물학적인 core” 확장 후보로 남긴다.

---

## 4.4 Shiu et al., Nature 2024

“A Drosophila computational brain model reveals sensorimotor processing”

FlyWire 기반 whole-brain-scale LIF model이다.

장점:
- 약 125k 이상의 뉴런, 50M 규모 synapse 수준의 whole-brain 계산 모델
- taste, feeding, grooming 등 sensorimotor processing을 보여준다.

하지만 현재 연구의 navigation/steering task와 직접 맞지 않으며, 첫 모델로는 지나치게 크다.

현재 연구는 “전뇌를 먼저 구현해야 한다”는 전제를 두지 않는다.

---

## 4.5 Mussells Pires et al., Nature 2024

Allocentric goal을 egocentric steering signal로 변환하는 central-complex pathway를 다룬다.

EPG heading + FC2 goal → PFL3 steering 구조로, Westeinde 모델과 독립적인 참고/대조 모델로 가치가 있다.

향후 Westeinde core 결과가 특정 모델에만 의존하는지 확인하기 위한 second-core replication에 적합하다.

---

## 4.6 FLYNN, arXiv 2026

초파리 connectome topology를 활용한 learned RNN 계열 robot navigation 연구.

중요한 의미:
- “fly topology를 robot navigation에 쓴 최초 연구” 같은 넓은 novelty claim은 이미 불가능하다.
- 우리 연구는 learned connectome-topology RNN과 차별화해야 한다.

우리 연구의 차별점 후보:
- trained RNN이 아니라 fixed biological/connectome-informed steering circuit
- body dynamics parameter를 체계적으로 변화
- body–brain mismatch tolerance/failure boundary 측정
- actual-motion feedback
- plugin의 역할을 goal-blind interface로 제한

---

## 4.7 Loihi 2 fly connectome 연구

FlyWire-scale network를 neuromorphic hardware에서 실행하는 방향의 연구.

현재 단계에서는 직접 사용하지 않는다.

의미:
- 장기적으로 neuromorphic implementation이 가능할 수 있다는 참고 근거
- 현재 첫 연구의 novelty나 구현 요구사항과는 별개

---

## 4.8 FlyGym / NeuroMechFly

향후 3D fly-like body 또는 physics-based embodiment를 구현할 때 사용할 수 있는 후보.

현재 첫 heading-recovery model에서는 필요하지 않다.

---

# 5. 첫 core로 사용할 모델

첫 core는 Westeinde et al. 2024의 reduced PFL2/PFL3 → DNa03 → DNa02 steering model을 기반으로 한다.

기준 connectome:
- hemibrain v1.2.1

주요 populations:
- PFL3R
- PFL3L
- PFL2
- DNa03R/L
- DNa02R/L

논문에서 사용한 reduced population simulation:
- PFL2 약 1000 units
- PFL3R 약 1000 units
- PFL3L 약 1000 units
- goal representation도 quasi-continuous population으로 표현

실제 생물학적 뉴런 수보다 훨씬 많은 discretization을 사용하는 이유는 neural phase space를 부드럽게 근사하기 위해서다.

---

# 6. Westeinde 모델에서 현재까지 확인한 핵심 요소

heading angle을 h라고 하고, neural space angle을 theta라 둘 때, 대표적인 입력은 다음 형태이다.

PFL3R:
I_PFL3R = cos(theta - theta0 - h + 67.5°)

PFL3L:
I_PFL3L = cos(theta - theta0 - h - 67.5°)

PFL2:
I_PFL2 = cos(theta - theta0 - h + 180°)

goal:
I_goal = A cos(theta_g - theta0 - h)

기본적으로 A = 1.

중요한 phase shifts:
- PFL3R: +67.5°
- PFL3L: -67.5°
- PFL2: 180°

connectivity 기반 상대 weight heuristic:
- PFL3 → DNa03 = 1
- PFL3 → DNa02 = 1
- PFL2 → DNa03 = 4
- DNa03 → DNa02 = 12

현재 첫 모델에서는 presynaptic pathway를 excitatory/cholinergic로 취급한다.

activation:
- normalization
- ELU-type nonlinear function

주의:
원 논문의 정확한 normalization과 notebook implementation은 반드시 공식 code와 재확인해야 한다.
논문 Methods만 보고 임의 구현한 후 “exact reproduction”이라고 주장하면 안 된다.

출력:
DNa02R - DNa02L

이 값이 turning tendency / rotational velocity와 연결된다.

논문 관계는 다음처럼 비례관계이다.

dtheta/dt ∝ DNa02R - DNa02L

즉 universal physical deg/s conversion constant가 직접 주어진 것은 아니다.

따라서 simulation에서는 별도의 decoder를 명시적으로 둔다.

r_brain = kappa * (DNa02R - DNa02L)

kappa는 neural core 내부 parameter와 구분한다.

kappa는 Stage 0 reproduction/calibration에서만 결정하고 이후 body experiment에서는 고정한다.

---

# 7. Stage 0의 정확한 목적

Stage 0는 “몸체를 붙이기 전에 neural steering core를 검증하는 단계”이다.

목표:
1. heading/goal input이 올바른 steering sign을 만드는지 확인
2. 좌우 symmetry 확인
3. PFL2 anti-goal behavior 확인
4. DNa03 indirect pathway와 DNa02 direct pathway 방향 확인
5. ideal closed-loop에서 목표 방향으로 수렴하는지 확인
6. source figure/trend와 qualitative 또는 가능한 범위에서 quantitative reproduction 확인

Stage 0가 실패하면 body simulation으로 넘어가지 않는다.

---

# 8. Stage 0 실패 시 진단 순서

baseline이 이상하면 아래 순서로 검사한다.

1. degree ↔ radian 오류
2. R/L sign 반전
3. PFL3 ±67.5° phase shift
4. PFL2 180° phase shift
5. 상대 weights 1:4:12
6. ELU와 normalization
7. gain S 적용 위치
8. decoder gain kappa
9. noise filtering
10. closed-loop update order

특히 body dynamics를 붙여서 문제를 가리면 안 된다.

---

# 9. 첫 행동 과제

첫 task는 “target approach”가 아니라 “goal heading recovery / stabilization”으로 고정한다.

이유:
- Westeinde core가 직접적으로 steering output을 제공한다.
- forward-speed generation을 별도로 가정할 필요가 없다.
- 가장 적은 인공 요소로 circuit function을 검증할 수 있다.
- body limitation의 효과를 순수하게 yaw dimension에서 분리할 수 있다.

목표 heading:
psi_g = 0

초기 heading error 후보:
±30°
±60°
±90°
±135°
170°

정확히 180°:
대칭성 때문에 steering output이 0이 되거나 불안정 평형점이 될 수 있으므로 별도 stress test로 둔다.

---

# 10. 첫 body model

첫 body는 특정 드론 모델이 아니다.

정확한 명칭:
“abstract 1-DOF yaw plant”

state:
x_body = (psi, r)

psi:
actual yaw angle

r:
actual yaw rate

brain command:
r_brain

plugin output:
r_cmd

body dynamics:
a_req = (r_cmd - r) / tau_r

a = clip(a_req, -alpha_max, +alpha_max)

r_{n+1} = clip(r_n + a * dt_body, -r_max, +r_max)

psi_{n+1} = wrap(psi_n + r_{n+1} * dt_body)

---

# 11. tau_r와 alpha_max의 의미

tau_r:
first-order response lag

즉 actuator/body가 명령된 yaw rate에 얼마나 빠르게 접근하는가를 나타낸다.

alpha_max:
hard yaw-acceleration limit

큰 command step이 들어올 때 물리적으로 허용되는 최대 rate-change를 제한한다.

둘은 서로 다른 개념이다.

- tau_r는 전반적 bandwidth / response time
- alpha_max는 큰 입력에서의 hard saturation

두 효과를 동시에 사용하는 것은 가능하다.

다만 논문/보고서에서 double counting처럼 보이지 않도록 이 차이를 명시해야 한다.

---

# 12. timestep과 실행 순서

Westeinde source model의 core update:
10 Hz

따라서:
T_core = 0.1 s

초기 abstract plant에서는 예를 들어:
dt_body = 0.0025 s

그러면 neural update 1회당 body integration 40회.

zero-order hold:
한 neural cycle 동안 r_cmd를 유지한다.

정확한 cycle:

1. body의 actual psi 읽기
2. sensor가 observed heading 생성
3. fly core 입력
4. DNa02R/L 계산
5. decoder가 r_brain 생성
6. plugin이 r_cmd 생성
7. body를 dt_body 단위로 여러 번 적분
8. actual psi, r 갱신
9. 다음 neural cycle에서 actual motion을 다시 sensor로 보냄

절대 command 자체를 actual heading으로 feedback하지 않는다.

---

# 13. sensor model

첫 실험에서는 privileged / ideal sensor를 사용할 수 있다.

theta_obs = psi_actual

단, 이것을 명확히 “ideal heading sensor” 또는 “privileged state observation”이라고 표시해야 한다.

ground-truth state는 평가용으로만 사용하는 것이 기본 원칙이다.

향후 확장:
- sensor noise
- latency
- dropout
- partial observation
- IMU-like model
- visual heading estimate

하지만 첫 본실험에서는 body constraint 효과를 분리하기 위해 이상적인 heading sensor가 적절하다.

---

# 14. plugin 설계

## 14.1 Plugin B — 단순 rate clip

r_cmd = clip(r_brain, -r_max, +r_max)

가장 단순한 body-side feasibility interface이다.

이 경우 acceleration limitation은 body model 자체가 처리한다.

---

## 14.2 Plugin C — state-aware / plant-aware interface

초기 아이디어는 다음과 같은 feasibility clamp였다.

lo = max(-r_max, r - alpha_max * tau_r)
hi = min(+r_max, r + alpha_max * tau_r)

r_cmd = clip(r_brain, lo, hi)

하지만 이 형태는 현재 first-order plant에서 body 내부 acceleration saturation과 거의 같은 효과를 만들어 B와 trajectory가 사실상 동일해질 가능성이 있다.

따라서 C는 본실험 전에 다시 설계해야 한다.

현재 추천 방향:
- goal-blind 유지
- heading error를 받지 않음
- r_brain과 actual r, body parameters만 사용
- future body response를 약하게 보상하는 reference governor 형태

예시 아이디어:

a_target = clip(
    (r_brain - r) / T_core,
    -alpha_max,
    +alpha_max
)

r_cmd = clip(
    r + tau_r * a_target,
    -r_max,
    +r_max
)

그러나 이것도 반드시 B와 실제로 다른 dynamics를 만드는지 수학적/수치적으로 검증한 후 채택해야 한다.

중요:
C가 B보다 좋아야만 연구가 성공하는 것은 아니다.

C가 효과가 없거나 오히려 악화되어도 연구 결과가 된다.

---

# 15. generic controller baseline D

단순 engineering baseline을 둔다.

예:
r_ref = -K_p * e

e = wrap(psi - psi_g)

K_p는 calibration set에서만 선택하고 test에서는 freeze한다.

역할:
- 동일한 body condition에서 task 자체가 가능한지 확인
- fly-derived circuit가 일반 controller보다 특별한 robust behavior를 보이는지 비교
- 실패가 body infeasibility 때문인지 neural controller 때문인지 분리

PID까지 처음부터 넣을 필요는 없다.
첫 baseline은 단순 P controller가 충분하다.

---

# 16. 기본 실험군

A. ideal body + fixed fly core

목적:
- neural core 자체의 baseline
- source closed-loop와 가장 가까운 조건

B. constrained body + simple rate clip

목적:
- body limitation이 core performance에 미치는 영향

C. same constrained body + state-aware plugin

목적:
- goal-blind body-side adaptation의 효과

D. same constrained body + simple general controller

목적:
- engineering baseline
- task feasibility sanity check

추가 ablation:

C-only / core removed
- core output을 0 또는 constant로 둠
- plugin이 goal을 모르기 때문에 스스로 navigation해서는 안 됨

feedback removal
- main condition이 아니라 causal ablation
- actual-motion feedback의 중요성을 확인

rewiring control
- connectome structure contribution을 주장할 경우에만 추가
- 첫 연구 필수는 아님

---

# 17. body parameter scan

초기에는 절대적인 “실제 드론 스펙”을 임의로 넣지 않는다.

먼저 ideal baseline에서 neural command distribution을 측정한다.

R99 = 99th percentile of |r_brain|

그 후 normalization한다.

r_max / R99 후보:
0.25
0.5
0.75
1.0
1.5

tau_r / T_core 후보:
0.25
0.5
1
2
4

T_core = 0.1 s 이므로 예를 들면:
25 ms
50 ms
100 ms
200 ms
400 ms

alpha_max / (R99 / T_core) 후보:
0.25
0.5
1
2
4

이 값들은 “실제 드론 사양”이라고 주장하지 않는다.

의미:
brain command timescale 대비 body bandwidth를 sweep하는 무차원화된 virtual actuator conditions.

---

# 18. 계산량

현재 reduced neural circuit + abstract body는 계산량이 매우 작다.

사용자 컴퓨터에서 직접 구현 가능하다.

예상:
- GPU 필요 없음
- 일반 CPU로 충분
- RAM 수 GB 이하
- 프로젝트 코드/결과는 수 GB도 필요하지 않음

3D visualization이나 MuJoCo를 붙이면 GPU 사용량이 증가할 수 있지만, 현재 연구의 neural computation 자체는 가볍다.

따라서 별도의 서버나 HPC를 전제로 하지 않는다.

---

# 19. 3D 그래픽 확장 — 중요 최신 사용자 결정

사용자는 최종적으로 시뮬레이션을 3D 그래픽으로 시각화하려 한다.

따라서 다른 채팅에서 구현할 때는 처음부터 향후 3D 연결이 가능한 architecture를 잡는 것이 좋다.

중요:
3D visualization과 physics simulation은 구분한다.

가능한 단계:

Stage A:
수치 계산 + matplotlib plot

Stage B:
간단한 3D visualizer
- body orientation
- target direction
- trajectory
- sensor rays
- neural command visualization

Stage C:
3D physics engine
- MuJoCo 등
- 실제 rigid-body dynamics
- contact / drag / inertia 등

처음부터 Stage C를 강제하지 않는다.

---

# 20. 3D 구현에서 추천되는 역할 분리

NeuralCore
→ 순수 neural computation

Decoder
→ neural output을 abstract action으로 변환

ConstraintPlugin
→ body capability에 맞는 command shaping

BodyModel
→ 실제 dynamics 계산

SensorModel
→ body/environment state에서 neural input 생성

Environment
→ target, landmark, obstacle, perturbation 정의

Renderer
→ simulation state를 3D로 보여줌

Evaluator
→ metric 계산

이 구조를 유지하면 renderer를 바꿔도 neural core가 영향을 받지 않는다.

---

# 21. 3D body를 처음부터 복잡한 드론으로 만들 필요가 없는 이유

첫 3D 시각화는 body를 단순한 arrow, capsule, small rigid object로 표현해도 된다.

연구의 핵심은 시각적 사실성이 아니라:

- neural core output
- body response
- actual feedback
- constraint-induced failure

이다.

따라서 3D object는 처음에는 상징적인 rigid body여도 충분하다.

나중에 필요하면:
- fly-like body
- quadrotor-like body
- wheeled robot
- underwater vehicle
등으로 교체 가능하다.

장기적으로 서로 다른 body를 비교하는 것이 이 연구의 중요한 확장 방향이다.

---

# 22. 2D/3D 확장 시 중요한 연구 질문

1-DOF yaw에서 성공하면 다음 단계는 planar navigation이다.

state 예:
x
y
psi
v
r

goal:
(x_g, y_g)

그러나 이때 neural core가 goal direction을 어떻게 받는지 명확히 정의해야 한다.

core가 직접 global position을 받아서는 안 된다.

가능한 구조:
environment
→ sensor / goal-vector preprocessing
→ desired heading representation
→ fly core
→ steering

즉 global coordinate를 biological core에 직접 꽂지 않는다.

---

# 23. obstacle avoidance는 첫 연구에서 넣지 않는다

현재 연구의 핵심은 body–brain mismatch이다.

obstacle avoidance를 넣으면 다음이 추가된다.

- path planning
- collision avoidance
- perception
- local/global planner
- target selection

이들은 별도 연구가 된다.

따라서 첫 연구에서는 목표 방향 회복과 유지에 집중한다.

---

# 24. noise와 disturbance

초기 실험:
noise = 0

이유:
body limitation 효과만 분리

그 후:
- neural noise
- sensor noise
- external disturbance
- delay

Westeinde source-like noise를 재현할 때:
- Gaussian noise
- low-pass filter
- source scaling
을 original implementation과 최대한 동일하게 맞춰야 한다.

paired seeds:
B와 C를 비교할 때 같은 noise seed 사용

통계 단위는 “time sample”이 아니라 independent trial이다.

---

# 25. 성공 기준

예시 분석 기준:

10초 안에
|e| <= 15°
영역에 들어가고,
최소 1초 이상 그 범위에 유지

단, 이것은 생물학적 threshold가 아니라 simulation analysis criterion이다.

따라서 sensitivity analysis:
10°
15°
20°
를 함께 볼 수 있다.

---

# 26. 주요 metric

필수:

1. success rate
2. settling time
3. integrated absolute error

IAE = ∫ |e(t)| dt

4. maximum overshoot
5. zero-crossing count / oscillation count
6. total rotation

∫ |r(t)| dt

7. brain-body mismatch

∫ |r_brain - r_actual| dt

8. plugin intervention fraction
9. rate saturation fraction
10. acceleration saturation fraction
11. timeout fraction
12. failure classification

---

# 27. 실패를 한 종류로 묶으면 안 된다

실패를 최소 다음으로 구분한다.

- body-infeasible
- near-infeasible
- slow but stable
- saturation dominated
- overshoot
- persistent oscillation
- controller instability
- sensor/feedback failure
- true neural-core failure

body-infeasible condition을 fly-core failure로 분류하면 안 된다.

예:
r_max가 너무 작아 물리적으로 10초 안에 120° 회전조차 불가능하다면 neural controller의 잘못이 아니다.

---

# 28. feasibility 판정

간단한 하한 계산 또는 numerical reachability check를 둔다.

최소한:
- r_max
- alpha_max
- timeout
- required angle

을 이용해 ideal bang-bang motion의 최소 시간을 추정한다.

tau_r가 들어가면 analytic bound가 복잡해질 수 있으므로 numerical reachability lower-bound를 별도 계산하는 것이 더 안전하다.

---

# 29. 가장 중요한 결과 형태

단순 trajectory 몇 개를 보여주는 것으로 끝내면 연구가 약하다.

최종적으로는 parameter space에서 다음과 같은 phase map을 만드는 것이 가장 좋다.

axes 예:
r_max
alpha_max
tau_r

classification:
- stable recovery
- slow recovery
- persistent saturation
- oscillatory recovery
- overshoot dominant
- failure
- body infeasible

핵심 결과는 다음 형태가 될 수 있다.

“fixed fly-derived steering core가 허용하는 body bandwidth/actuator constraint의 경계”

이것이 첫 논문의 중심 결과 후보이다.

---

# 30. 연구가 너무 당연한 결과로 끝나는 것을 피하는 방법

“body가 느리면 성능이 나빠진다”만으로는 약하다.

따라서 반드시 다음을 분석한다.

- 어느 parameter가 먼저 failure를 유발하는가?
- rate limit과 lag가 같은 failure를 만드는가?
- acceleration saturation과 response lag가 다른 dynamical signature를 보이는가?
- feedback lag 때문에 oscillation이 생기는가?
- neural command bandwidth와 plant bandwidth 사이 특정 ratio에서 transition이 있는가?
- same body에서 P controller와 fly circuit failure boundary가 다른가?

즉 성능 감소가 아니라 failure mechanism이 핵심이다.

---

# 31. negative result의 가치

이 연구는 fly circuit가 반드시 일반 controller보다 좋아야 성공하는 연구가 아니다.

가능한 결과 A:
fly circuit가 넓은 body constraint 범위에서도 안정

→ biological computation의 robustness

가능한 결과 B:
특별히 강건하지 않음

→ connectome-derived steering circuit에도 최소 body–brain timescale matching이 필요

가능한 결과 C:
simple plugin이 큰 효과 없음

→ external interface adaptation만으로는 embodiment mismatch를 해결하기 어려움

가능한 결과 D:
generic P controller가 훨씬 강함

→ biological core의 장점은 raw robustness가 아니라 다른 특징에 있을 가능성

모두 연구 결과가 된다.

---

# 32. novelty 관련 주의

다음은 novelty로 주장하면 안 된다.

- 최초의 fly navigation model
- 최초의 fly closed-loop steering
- 최초의 connectome-inspired robot control
- actuator clipping을 최초 도입

이미 관련 선행연구가 존재한다.

현재 가능한 novelty 후보:
- fixed fly-derived steering circuit의 systematic embodiment-constraint sweep
- body–brain timescale mismatch phase diagram
- actual-motion feedback를 포함한 cross-body tolerance 분석
- goal-blind plugin이 fixed biological core의 portability를 얼마나 개선하는지 평가

그러나 “최초” 주장은 더 넓은 systematic literature review 후에만 사용한다.

---

# 33. 데이터셋/버전 관리에서 반드시 지킬 사항

Westeinde first core:
hemibrain v1.2.1

다른 dataset의 neuron ID를 섞지 않는다.

특히 FlyWire v783 DNa02 ID 등을 hemibrain 기반 첫 모델에 직접 섞으면 안 된다.

나중에 FlyWire로 업그레이드한다면:
PFL2
PFL3
DNa03
DNa02

모두 동일 FlyWire version 내에서 다시 매핑해야 한다.

---

# 34. provenance 기록

각 run마다 최소 다음을 저장한다.

- source paper DOI
- connectome dataset/version
- code version / git commit
- core parameter set
- neural population discretization
- activation function
- normalization
- phase shifts
- connection weights
- decoder gain
- core timestep
- body timestep
- plugin type
- body parameters
- initial condition
- seed
- sensor configuration
- success criterion
- timeout

---

# 35. 추천 로컬 프로젝트 구조

처음부터 지나치게 복잡하게 만들 필요는 없지만, 3D 확장을 고려하면 다음 정도가 적절하다.

drosophila_nav/
│
├─ src/
│  ├─ core/
│  │  └─ westeinde2024.py
│  ├─ decoder/
│  │  └─ steering.py
│  ├─ plugins/
│  │  ├─ passthrough.py
│  │  ├─ rate_clip.py
│  │  └─ state_governor.py
│  ├─ body/
│  │  ├─ yaw_plant.py
│  │  └─ planar_body.py
│  ├─ sensors/
│  │  └─ ideal_heading.py
│  ├─ environment/
│  │  └─ heading_task.py
│  ├─ sim/
│  │  └─ closed_loop.py
│  ├─ render/
│  │  └─ viewer3d.py
│  └─ eval/
│     └─ metrics.py
│
├─ experiments/
│  ├─ stage0_reproduction.py
│  ├─ stage1_yaw_constraints.py
│  └─ stage2_3d_demo.py
│
├─ configs/
│  ├─ reproduction.yaml
│  ├─ body_scan.yaml
│  └─ evaluation.yaml
│
├─ provenance/
│  ├─ literature.yaml
│  ├─ model_version.yaml
│  └─ reproduction_adjustments.yaml
│
├─ tests/
│  ├─ test_core_symmetry.py
│  ├─ test_body_limits.py
│  ├─ test_plugin_limits.py
│  ├─ test_feedback.py
│  └─ test_reproducibility.py
│
└─ results/

---

# 36. 사용자의 로컬 구현 방향

중요한 최신 결정:
코드는 ChatGPT 환경에서 대신 완성하는 것이 아니라 사용자의 PC에서 직접 구현한다.

다음 채팅의 역할:
- 설치 안내
- 파일 생성 순서
- 각 파일의 코드 제공
- 실행 명령
- 예상 output
- 오류 메시지 해석
- 결과 검증

사용자는 직접 코드를 실행하고 나온 결과를 다시 전달한다.

그 결과를 보고 다음 단계로 진행한다.

---

# 37. 처음 필요한 소프트웨어

최소:
- Python
- numpy
- scipy
- matplotlib
- pandas
- pyyaml

3D visualization 후보:
- matplotlib 3D
- PyVista
- VPython
- Panda3D
- pyglet/OpenGL
- MuJoCo viewer

실제 rigid-body physics가 필요할 때:
- MuJoCo

처음부터 MuJoCo를 필수로 두지 않는다.

---

# 38. 3D 시각화 도구 선택 원칙

가장 중요한 것은 연구 core와 renderer를 분리하는 것이다.

선택 기준:

matplotlib 3D:
- 가장 단순
- 연구 결과 plot에 좋음
- 실시간 interactive visualization은 제한적

VPython:
- 매우 빠르게 3D 움직임 시각화 가능
- 초보자 친화적
- 물리 engine 자체는 아님

PyVista:
- scientific visualization에 강함
- trajectory/vector/mesh 표현 좋음

Panda3D:
- custom 3D application에 강함
- 구현량 증가

MuJoCo:
- physics + 3D visualization
- rigid-body simulation에 적합
- 나중 단계 추천

첫 구현에서는 visualization만 필요하면 VPython 또는 PyVista가 더 간단할 수 있다.

실제 동역학까지 3D로 올릴 때 MuJoCo로 확장 가능하다.

---

# 39. 현재 연구에서 3D가 가지는 의미

3D 그래픽은 단순 장식이 아니라 다음에 도움이 될 수 있다.

- actual heading vs neural desired turn 시각화
- body lag를 직관적으로 표현
- yaw saturation 표시
- sensor direction 시각화
- target direction 표시
- trajectory와 overshoot 확인
- body 종류를 바꿔도 같은 core를 쓰는 장기 데모

하지만 metric은 3D 화면을 보고 수동 평가하지 않는다.

항상 numerical log를 별도로 저장한다.

---

# 40. 첫 3D scene 추천

환경:
- 평면 ground
- body 하나
- goal direction marker
- heading arrow
- neural command arrow
- actual angular velocity indicator

body:
- 간단한 arrow/capsule/triangular craft

첫 task:
- 위치는 고정
- yaw만 회전
- goal heading 0°
- 초기 yaw를 여러 값으로 설정

즉 1-DOF yaw experiment도 3D로 충분히 보여줄 수 있다.

그 후 planar translation을 추가한다.

---

# 41. 처음부터 하지 말아야 할 것

다음은 첫 구현에서 피한다.

- whole-brain simulation
- FlyWire 전체 140k neuron 직접 구동
- 복잡한 spiking neuron model
- 실제 quadrotor aerodynamics
- obstacle avoidance
- SLAM
- path planning
- reinforcement learning
- per-body retraining
- real-world sensor model
- hardware control
- chip implementation

이들은 후속 확장이다.

---

# 42. 첫 실제 구현 순서

## Phase 0 — 환경 구축

1. Python environment
2. git repository
3. numpy/scipy/matplotlib 설치
4. 폴더 구조 생성
5. config/provenance 파일 생성

## Phase 1 — core 단독

1. neural phase grid 생성
2. heading representation
3. goal representation
4. PFL3R/PFL3L/PFL2 activity
5. normalization + ELU
6. DNa03
7. DNa02
8. steering output
9. symmetry plot
10. heading-error vs neural output curve

## Phase 2 — ideal closed loop

1. decoder 추가
2. ideal body:
   psi_{n+1} = psi_n + r_brain * T_core
3. heading recovery 확인
4. source trend와 비교
5. Stage 0 gate

## Phase 3 — abstract body

1. yaw plant 추가
2. actual yaw rate state 추가
3. rate limit
4. acceleration limit
5. response lag
6. actual-motion feedback

## Phase 4 — B condition

simple clip plugin

## Phase 5 — parameter sweep

r_max
alpha_max
tau_r

## Phase 6 — failure classification

success / infeasible / oscillation / saturation 등

## Phase 7 — plugin C

B와 실제 dynamics 차이가 있는지 먼저 검증

## Phase 8 — engineering baseline D

P controller

## Phase 9 — 3D viewer

수치 simulation은 그대로 두고 renderer만 연결

---

# 43. Stage 0에서 반드시 저장할 그래프

1. steering output vs heading error
2. PFL3R activity
3. PFL3L activity
4. PFL2 activity
5. DNa02R/L output
6. closed-loop heading trajectory
7. ±same-angle symmetry comparison

가능하면 original paper의 relevant figure/trend와 나란히 검토한다.

---

# 44. test code로 확인해야 할 최소 항목

1. angle wrap correctness
2. radians/degrees consistency
3. left-right symmetry
4. zero error near zero steering
5. opposite errors produce opposite steering
6. output finite
7. rate clip respected
8. acceleration cap respected
9. actual motion used as feedback
10. reproducible result with fixed seed
11. same core params across body conditions
12. plugin cannot navigate with zero core output

---

# 45. 연구 진행 중 가장 중요한 질문

항상 아래를 확인한다.

“지금 나타난 효과가 neural core 때문인가, decoder 때문인가, plugin 때문인가, body dynamics 때문인가, sensor 때문인가?”

이를 분리하지 못하면 결과 해석이 약해진다.

그래서 각 module을 독립 파일로 나누고 ablation을 둔다.

---

# 46. decoder calibration 주의

Westeinde model의 DNa02 difference는 절대적인 deg/s가 아니다.

따라서 physical-like body에 연결하려면 scale factor kappa가 필요하다.

원칙:
- Stage 0 calibration에서만 정한다.
- calibration rule을 문서화한다.
- 이후 모든 body experiment에서 freeze한다.
- body condition마다 다시 tuning하지 않는다.

가능한 calibration criterion:
- original source closed-loop timescale
- 특정 heading error에서 source-like angular velocity
- ideal model settling time

정확한 기준은 official notebook 확인 후 확정한다.

---

# 47. initial conditions와 trial design

paired design을 권장한다.

같은:
- initial heading
- goal
- seed
- disturbance

를 A/B/C/D에 모두 사용한다.

그래야 condition difference를 직접 비교할 수 있다.

calibration set과 test set도 분리한다.

예:
calibration:
±30
±60
±90
±135

test:
±45
±75
±120
±150
±170

정확한 구성은 implementation 시 조정 가능하나, test condition을 calibration에 반복 사용하지 않는 것이 바람직하다.

---

# 48. 180° condition

정확한 anti-goal 180°는 특별 취급한다.

이유:
sinusoidal symmetric representation에서 좌우 turning tendency가 정확히 상쇄될 수 있다.

따라서 180°에서 움직이지 않는다고 즉시 controller failure로 부르지 않는다.

별도:
symmetry-breaking noise
small perturbation
±179°
등을 이용해 stress test 가능.

---

# 49. 외부 disturbance 확장

본 연구가 진행된 후 다음을 넣을 수 있다.

- sudden yaw impulse
- constant bias torque
- sensor delay
- heading observation noise
- transient signal dropout

목적:
body limitation뿐 아니라 feedback robustness 분석

하지만 첫 main sweep과 섞지 않는다.

---

# 50. 이후 cross-body 연구

장기적으로 동일 core를 서로 다른 body에 연결한다.

예:
- fast yaw / low inertia body
- slow yaw / high inertia body
- ground robot
- aerial body
- underwater-like body

각 body는 own plugin을 가진다.

핵심:
core는 공통
body-specific feasibility layer만 교체

이 단계에서 처음 장기 비전인 “common navigation core”를 직접 검증할 수 있다.

---

# 51. neuromorphic / chip 방향

현재 연구에는 포함하지 않는다.

장기적으로:
- reduced circuit compression
- spiking conversion
- neuromorphic deployment
- Loihi-class hardware
- FPGA/ASIC

등을 검토할 수 있다.

그러나 첫 연구에서 hardware efficiency를 결과처럼 미리 주장하지 않는다.

biology
control performance
hardware efficiency

는 별도 claim이다.

---

# 52. 보고서에서 구분해야 할 세 종류의 주장

1. 생물학적 주장
“이 model이 Drosophila central-complex steering pathway를 기반으로 한다.”

2. control 주장
“이 fixed circuit가 특정 body constraint 범위에서 heading recovery를 유지한다.”

3. engineering/hardware 주장
“이 구조가 실제 robot/chip에 유용하다.”

현재 연구에서 직접 강하게 검증할 수 있는 것은 주로 2번이다.

1번은 선행연구에 근거한다.
3번은 장기 가능성으로만 다룬다.

---

# 53. 현재 연구의 가장 강한 한 문장

“초파리 central-complex에서 유래한 고정 steering circuit를 서로 다른 운동 제약을 가진 가상 몸체에 폐루프로 연결하고, actual-motion feedback 하에서 body–brain dynamics mismatch가 행동 안정성에 미치는 영향을 체계적으로 지도화한다.”

---

# 54. 연구 제목 후보

한국어:
“초파리 유래 항법 회로의 몸체 동역학 제약에 대한 폐루프 강건성 분석”

또는

“초파리 항법 회로와 몸체 제약 플러그인의 폐루프 시뮬레이션 연구”

영문:
“Closed-Loop Robustness of a Drosophila-Derived Steering Circuit under Embodiment Constraints”

또는

“Embodiment Tolerance of a Fixed Drosophila-Derived Navigation Circuit in Closed-Loop Simulation”

최종 제목은 결과를 보고 정한다.

---

# 55. 연구의 현재 상태

현재 상태는 “주제 검토 단계”를 통과한 상태다.

다음 단계는 실제 구현이다.

다만 중요한 최신 수정:
ChatGPT가 자체 환경에서 코드를 완성해 사용자에게 결과 zip을 주는 방식이 아니라, 사용자가 자신의 컴퓨터에서 직접 구현하는 방식으로 진행한다.

또한 사용자는 3D 그래픽으로 simulation을 보여주고 싶어 한다.

따라서 다음 채팅에서는:
1. 로컬 개발환경
2. 첫 repository
3. Stage 0 core
4. 3D viewer와 연결 가능한 architecture

순으로 시작해야 한다.

---

# 56. 다음 채팅이 가장 먼저 해야 할 일

전체 연구계획을 다시 설명하지 말고 아래부터 시작한다.

STEP 1:
사용자 PC에서 Python project 생성

STEP 2:
최소 dependency 설치

STEP 3:
folder structure 생성

STEP 4:
westeinde2024.py 구현

STEP 5:
core symmetry / I-O plot 확인

STEP 6:
ideal closed loop

STEP 7:
사용자가 실행 결과를 보여주면 diagnostic

3D renderer는 core 검증을 방해하지 않도록 별도 module로 준비한다.

---

# 57. 다음 채팅에 요구되는 작업 스타일

- 하나씩 실제 구현
- 코드를 사용자가 복사/저장/실행 가능하게 제공
- 파일명을 명확히 지정
- 실행 명령 정확히 제시
- 예상 출력 제시
- 오류가 나면 로그를 보고 수정
- 임의로 새 연구 방향으로 변경하지 않기
- 사용자가 확정한 것과 assistant 제안을 구분
- source-derived parameter와 arbitrary simulation parameter를 구분
- 실제 값을 모르면 만들지 않기
- body parameter를 실제 드론 스펙이라고 속이지 않기

---

# 58. 최종 우선순위

우선순위 1:
Westeinde reduced steering core의 정확한 reproduction

우선순위 2:
ideal closed loop

우선순위 3:
abstract yaw body + actual-motion feedback

우선순위 4:
A/B robustness sweep

우선순위 5:
failure classification / phase map

우선순위 6:
plugin C와 P-controller baseline

우선순위 7:
3D visualization

우선순위 8:
planar/3D locomotion

우선순위 9:
second biological core / whole-brain expansion

---

# 59. 반드시 기억해야 할 경계

이 연구의 첫 성공 기준은 “초파리 뇌 전체를 가상 드론에 넣었다”가 아니다.

첫 성공 기준은 다음이다.

“생물학적으로 근거가 있는 고정 steering core를 폐루프로 실행하고, body dynamics가 달라질 때 기능 유지/실패의 구조를 재현성 있게 측정했다.”

이 결과가 확실하면 그 다음 연구로 확장할 수 있다.

---

# 60. 다음 작업자에게 주는 최종 지시

이 문서를 최신 기준으로 삼아 후속 작업을 진행하라.

전체 목표를 다른 주제로 바꾸지 말 것.
whole-brain 구현을 첫 단계로 강요하지 말 것.
obstacle avoidance나 RL을 임의로 추가하지 말 것.
body plugin이 navigation을 대신하지 않도록 할 것.
actual motion feedback를 반드시 유지할 것.
source model, simulation assumption, calibration parameter를 구분할 것.
user가 직접 local PC에서 구현하도록 안내할 것.
3D visualization을 염두에 둔 modular architecture를 사용할 것.

첫 구현은 “Westeinde-derived steering core → ideal closed loop → abstract yaw body” 순서로 시작한다.

그리고 Stage 0 reproduction이 확인되기 전에는 body robustness 결과를 해석하지 않는다.

---

# 핵심 요약

연구 질문:
고정된 초파리 유래 steering circuit가 body dynamics가 달라져도 기능을 유지하는가?

첫 core:
Westeinde et al. 2024 reduced PFL2/PFL3–DNa03–DNa02 steering model

첫 task:
goal-heading recovery

첫 body:
1-DOF yaw plant

핵심 변수:
r_max
alpha_max
tau_r

핵심 비교:
A ideal
B constrained + clip
C constrained + goal-blind state-aware plugin
D generic P controller

핵심 결과:
embodiment robustness/failure phase map

중요 원칙:
actual motion feedback
core frozen
plugin goal-blind
body infeasibility separate classification
source model and arbitrary simulation assumptions separated

구현:
사용자 자신의 PC에서 Python으로 직접 구현
GPU/HPC 필수 아님
향후 3D visualization 포함
renderer와 simulation core 분리

다음 즉시 작업:
로컬 프로젝트 생성 → Westeinde core 구현 → Stage 0 reproduction → ideal closed loop → body model → sweep → 3D viewer
