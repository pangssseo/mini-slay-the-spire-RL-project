# Mini Slay the Spire 강화학습 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 카드 기반 로그라이크 게임인 **Slay the Spire**의 전투 시스템을 단순화하여, 강화학습 에이전트가 턴제 카드 전투에서 어떤 행동을 선택해야 하는지 학습하도록 설계한 프로젝트입니다.

플레이어는 매 턴 제한된 에너지 안에서 카드를 선택하고, 적인 **Nibbit(깨작이)** 은 정해진 패턴에 따라 공격, 방어도 획득, 힘 증가 행동을 반복합니다.

본 프로젝트의 목표는 주어진 전투 상태에서 에이전트가 공격, 방어, 취약 부여, 턴 종료 중 적절한 행동을 선택하여 전투 성능을 향상시키는 것입니다.

비교한 에이전트는 다음과 같습니다.

* Random Agent
* Q-Learning Agent
* DQN Agent

---

## 2. 프로젝트 목표

본 프로젝트의 핵심 목표는 다음과 같습니다.

1. Slay the Spire 스타일의 단순화된 전투 환경을 직접 설계한다.
2. 강화학습 관점에서 State, Action, Reward를 정의한다.
3. Random, Q-Learning, DQN 알고리즘을 동일한 환경에서 비교한다.
4. 난이도별 실험을 통해 각 알고리즘의 성능 차이를 분석한다.
5. Action Timeline과 HP Trace를 통해 학습된 정책의 행동 패턴을 해석한다.

---

## 3. 환경 설명: Mini Nibbit Battle

전투는 플레이어와 적 Nibbit 사이의 1대1 턴제 전투로 구성됩니다.

플레이어는 매 턴 기본적으로 에너지 3을 가지고, 손패에 있는 카드 중 하나를 선택하여 사용합니다.
카드를 사용하거나 턴을 종료하는 것이 에이전트의 행동입니다.

적 Nibbit은 3턴 주기의 고정 패턴을 반복합니다.

### Nibbit 행동 패턴

| 패턴 턴 | 행동                 |
| ---: | ------------------ |
|   1턴 | 12 피해 공격           |
|   2턴 | 6 피해 공격 + 방어도 5 획득 |
|   3턴 | 힘 2 획득             |
|   반복 | 다시 1턴 패턴부터 반복      |

힘이 증가하면 이후 공격 피해량이 증가합니다.

예를 들어 힘이 2인 상태에서 12 피해 공격을 수행하면 실제 공격 피해는 14가 됩니다.

---

## 4. 난이도 설정

실험은 세 가지 난이도에서 수행했습니다.

| Mode   | Player HP | Nibbit HP | 설명                      |
| ------ | --------: | --------: | ----------------------- |
| Easy   |        80 |        46 | 플레이어 체력이 높아 가장 쉬운 조건    |
| Normal |        64 |        46 | 플레이어 체력을 낮춘 조건          |
| Hard   |        64 |        48 | 플레이어 체력 감소 + 적 체력 증가 조건 |

Easy에서 Normal로 갈수록 플레이어의 생존성이 낮아지고, Hard에서는 적의 체력이 증가하여 더 긴 전투와 더 정교한 행동 선택이 요구됩니다.

---

## 5. State 설계

에이전트는 현재 전투 상태를 관찰한 뒤 행동을 선택합니다.

State에는 다음 정보가 포함됩니다.

* 플레이어 HP
* 적 HP
* 적의 공격 의도
* 적의 방어도 획득 의도
* 적의 힘 증가 의도
* 적의 현재 방어도
* 적의 현재 힘
* 적의 취약 상태 지속 턴
* 플레이어의 현재 에너지
* 현재 손패의 Strike / Defend / Bash 카드 수
* 드로우 덱의 Strike / Defend / Bash 카드 수
* 버린 카드 더미의 Strike / Defend / Bash 카드 수

이러한 State 설계를 통해 에이전트는 단순히 현재 HP만 보는 것이 아니라, 적의 다음 행동, 현재 손패, 남은 덱 구성까지 고려할 수 있습니다.

---

## 6. Action 설계

에이전트가 선택할 수 있는 행동은 다음 네 가지입니다.

| Action ID | 행동       |
| --------: | -------- |
|         0 | Strike   |
|         1 | Defend   |
|         2 | Bash     |
|         3 | End Turn |

### 카드 효과

| 카드       | 비용 | 효과                  |
| -------- | -: | ------------------- |
| Strike   |  1 | 적에게 6 피해            |
| Defend   |  1 | 플레이어 방어도 5 획득       |
| Bash     |  2 | 적에게 8 피해 + 취약 2턴 부여 |
| End Turn |  0 | 현재 턴 종료             |

Bash는 단순 공격 카드가 아니라 적에게 취약 상태를 부여합니다.
취약 상태의 적은 이후 공격에서 더 큰 피해를 받기 때문에, Bash는 장기적인 공격 효율을 높이는 전략 카드로 설계되었습니다.

---

## 7. Reward 설계

Reward는 공격 효율과 생존성을 함께 반영하도록 설계했습니다.

| 상황           |        Reward |
| ------------ | ------------: |
| 적에게 피해를 줌    |       +damage |
| 플레이어가 피해를 받음 | -damage_taken |
| 전투 승리        |          +100 |
| 전투 패배        |          -100 |
| 불가능한 행동 선택   |            -5 |

이 Reward 설계를 통해 에이전트는 단순히 공격만 하는 것이 아니라, 전투에서 승리하고 피해를 줄이는 방향으로 학습할 수 있습니다.

---

## 8. 사용한 알고리즘

본 프로젝트에서는 세 가지 에이전트를 비교했습니다.

### 8.1 Random Agent

Random Agent는 현재 state를 고려하지 않고 가능한 행동 중 하나를 무작위로 선택합니다.
학습 기반 에이전트와 비교하기 위한 baseline으로 사용했습니다.

### 8.2 Q-Learning Agent

Q-Learning은 이산화된 state와 action에 대해 Q-table을 학습하는 방식입니다.

장점은 구현과 해석이 단순하다는 점입니다.
하지만 state 공간이 커질수록 Q-table이 커지고, 복잡한 state를 일반화하는 데 한계가 있습니다.

### 8.3 DQN Agent

DQN은 Q-table 대신 Neural Network를 사용하여 Q-value를 근사합니다.

State vector를 입력으로 받아 각 action의 가치를 출력하며, 복잡한 state 정보를 더 유연하게 활용할 수 있습니다.
본 프로젝트에서는 HP, 적 intent, 취약 상태, 힘, 덱 구성 등 다양한 정보를 포함하는 state를 사용했기 때문에 DQN과 Q-Learning의 차이를 비교하기에 적합합니다.

---

## 9. 실행 방법

### 9.1 패키지 설치

```bash
pip install numpy matplotlib torch
```

### 9.2 빠른 테스트 실행

```bash
python run_experiments.py --quick
```

이 명령어는 코드가 정상적으로 동작하는지 확인하기 위한 빠른 테스트입니다.

### 9.3 전체 실험 실행

```bash
python run_experiments.py
```

전체 실험은 Easy, Normal, Hard 세 난이도에서 Random, Q-Learning, DQN을 모두 실행합니다.
기본 학습 episodes 설정은 Q-learning 3,000회, DQN 1,500회이며 평가 episodes는 200회씩 수행
'9.4'를 통해 episodes를 커스터마이징하여 수행할 수 있다.

### 9.4 실험 시간을 줄여 실행(episodes 제한)

```bash
python run_experiments.py --q-episodes 1000 --dqn-episodes 1000 --eval-episodes 100
```

### 9.5 학습된 정책 확인

```bash
python inspect_policy.py --mode easy --agent dqn
python inspect_policy.py --mode normal --agent dqn
python inspect_policy.py --mode hard --agent dqn
```

Q-Learning 또는 Random Agent의 정책도 확인할 수 있습니다.

```bash
python inspect_policy.py --mode easy --agent qlearning
python inspect_policy.py --mode easy --agent random
```

### 9.6 Manual State 테스트

```bash
python manual_state.py
```

이 기능을 사용하면 특정 state를 직접 입력하고, 해당 state에서 action을 선택했을 때 reward와 next state가 어떻게 변하는지 확인할 수 있습니다.

---

## 10. 결과 파일 설명

실험을 실행하면 `outputs/` 폴더에 결과가 저장됩니다.

### 전체 결과 파일

| 파일                                 | 설명                        |
| ---------------------------------- | ------------------------- |
| `summary_all_modes.csv`            | 난이도별, 에이전트별 성능 요약         |
| `win_rate_by_mode.png`             | 난이도별 승률 비교                |
| `random_win_rate_by_mode.png`      | Random Agent의 난이도별 승률     |
| `average_reward_by_mode.png`       | 평균 Reward 비교              |
| `remaining_hp_by_mode.png`         | 평균 남은 HP 비교               |
| `random_reward_curves_by_mode.png` | Random Agent reward curve |

### 난이도별 세부 결과

각 난이도 폴더에는 Random, Q-Learning, DQN의 action timeline과 HP trace가 저장됩니다.

예시:

```text
outputs/easy/random_policy_action_timeline.png
outputs/easy/qlearning_policy_action_timeline.png
outputs/easy/dqn_policy_action_timeline.png

outputs/easy/random_policy_hp_trace.png
outputs/easy/qlearning_policy_hp_trace.png
outputs/easy/dqn_policy_hp_trace.png
```

Action Timeline은 각 turn에서 에이전트가 선택한 행동 sequence를 보여줍니다.
HP Trace는 turn 단위로 플레이어 HP와 Nibbit HP가 어떻게 변화하는지 보여줍니다.

---

## 11. 실험 결과 분석 요약

본 프로젝트에서는 Easy, Normal, Hard 세 난이도에서 Random, Q-Learning, DQN을 비교했습니다.

Random Agent는 state를 고려하지 않고 행동하기 때문에 reward curve와 action timeline에서 일관된 전략을 보이지 않습니다.
반면 Q-Learning과 DQN은 반복 학습을 통해 적의 패턴과 카드 효과를 반영한 행동을 선택할 수 있습니다.

Q-Learning은 단순하고 해석이 쉬운 장점이 있지만, state가 복잡해질수록 표현력이 제한될 수 있습니다.
DQN은 neural network를 사용하여 복잡한 state vector를 처리할 수 있으므로, HP, enemy intent, vulnerable, strength, deck state와 같은 다양한 정보를 더 유연하게 활용할 수 있습니다.

실험 결과는 다음 관점에서 분석했습니다.

* Win Rate
* Mean Reward
* Remaining HP
* Mean Turns
* Training Reward Curve
* Action Timeline
* Turn-level HP Trace

---

## 12. Q-Learning과 DQN 비교

Q-Learning과 DQN의 핵심 차이는 Q-value를 저장하고 표현하는 방식입니다.

| 항목          | Q-Learning         | DQN                           |
| ----------- | ------------------ | ----------------------------- |
| Q-value 표현  | Q-table            | Neural Network                |
| State 처리    | 이산화된 state 사용      | 연속적인 state vector 사용 가능       |
| 장점          | 단순하고 해석이 쉬움        | 복잡한 state에 더 유연함              |
| 단점          | state 공간이 커지면 비효율적 | 학습 불안정성과 hyperparameter 영향 존재 |
| 본 프로젝트 내 역할 | 기본 학습 알고리즘         | 복잡한 state 처리용 심층 강화학습 알고리즘    |

본 프로젝트에서는 두 알고리즘을 동일한 Mini Nibbit 환경에서 비교하여, 복잡한 state를 가진 카드 전투 환경에서 Q-Learning과 DQN이 어떤 차이를 보이는지 확인했습니다.

---

## 13. 보고서

최종 발표 보고서는 `report/` 폴더에 포함되어 있습니다.

```text
report/mini_sts_designed_graph_report.pptx
```

보고서에는 프로젝트 목표, 환경 설계, State/Action/Reward 정의, 알고리즘 비교, 실험 결과 그래프, 정책 행동 패턴 분석, 결론 및 한계가 포함되어 있습니다.

---

## 14. 한계 및 향후 과제

본 프로젝트는 실제 Slay the Spire의 전체 게임 구조를 단순화한 환경입니다.

현재 포함하지 않은 요소는 다음과 같습니다.

* 전투 후 카드 보상 선택
* 덱 성장
* 유물 효과
* 이벤트 선택
* 경로 선택
* 다중 적 전투
* 다양한 적 패턴 등

향후에는 카드 선택과 덱빌딩 요소를 추가하여, 단일 전투 최적화가 아니라 장기적인 덱 성장 전략을 학습하는 문제로 확장할 수 있습니다.

---

## 15. 프로젝트 정보

* 과목명: 강화학습
* 프로젝트명: 강화학습을 활용한 Mini Slay the Spire 전투 에이전트 최적화
* GitHub Repository: https://github.com/pangssseo/mini-slay-the-spire-RL-project
