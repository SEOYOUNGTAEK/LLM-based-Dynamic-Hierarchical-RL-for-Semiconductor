# llm_agent_ablation.py
# Ablation 실험용 에이전트 클래스 모음
#
# 1. RandomRuleAgent    — LLM 대신 랜덤 룰 생성 (Ablation 2: LLM vs Random Rule)
# 2. SoftRewardAdapter  — 룰을 action mask 대신 reward penalty로 변환 (Ablation 1: Hard vs Soft)
# 3. RandomMaskAgent    — 조건 없이 매 에피소드 랜덤 액션 차단 (Ablation 4: LLM vs Random Mask)

import random
import numpy as np
from typing import List, Dict, Any, Optional

from llm_agent import (
    LLMStrategicAnalyst, SafetyRule, sanitize_rules,
    METRIC_STEP_RANGE, URGENT_FORCE_THRESHOLD, TARGET_CONSTRAINT
)
from logger import Logger


# ──────────────────────────────────────────────────────────────
# Ablation 1 보조: Soft Reward Adapter
# train_ablation_soft.py 에서 사용.
# safety_agent가 생성한 현재 룰을 받아, 특정 action이 룰을 위반했을 때
# reward penalty를 반환한다 (action mask 대신).
# ──────────────────────────────────────────────────────────────

SOFT_PENALTY_PER_VIOLATION = -5.0  # lot_move_reward(~2.0/step)의 2.5배


class SoftRewardAdapter:
    """
    LLM 룰을 action mask 대신 reward penalty로 변환하는 어댑터.
    동일한 LLMStrategicAnalyst + SimpleSafetyAgent를 그대로 쓰되,
    훈련 루프에서 mask 대신 이 어댑터를 통해 penalty를 reward에 더한다.
    """

    def __init__(self, penalty: float = SOFT_PENALTY_PER_VIOLATION):
        self.penalty = penalty
        self.logger = Logger().get_logger()

    def get_penalty(self, chosen_action: int, state_summary: str, rules: list) -> float:
        """
        현재 rules와 state_summary를 기준으로, chosen_action이 룰을 위반하면
        위반 횟수 × penalty 반환. 위반 없으면 0.0.
        """
        if not rules:
            return 0.0

        # state_summary 파싱 (simple_safety_agent와 동일 로직)
        import re
        try:
            priority_match = re.search(r"urgent_priority_lot_count.*?: (\d+)", state_summary)
            shortage_match = re.search(r"production_shortage_pct.*?: ([\d.]+)", state_summary)
            iplt_match     = re.search(r"iplt_over_lot_count.*?: (\d+)", state_summary)
            metrics = {
                "urgent_priority_lot_count": int(priority_match.group(1)) if priority_match else 0,
                "production_shortage_pct":   float(shortage_match.group(1)) if shortage_match else 0.0,
                "iplt_over_lot_count":       int(iplt_match.group(1)) if iplt_match else 0,
            }
        except Exception:
            return 0.0

        total_penalty = 0.0
        for rule in rules:
            mv = metrics.get(rule.metric, 0)
            op = rule.operator
            th = rule.threshold
            triggered = (
                (op == ">=" and mv >= th) or
                (op == ">"  and mv >  th) or
                (op == "<=" and mv <= th) or
                (op == "<"  and mv <  th) or
                (op == "==" and mv == th)
            )
            if triggered and chosen_action in rule.forbidden_actions:
                total_penalty += self.penalty
                self.logger.debug(
                    f"[SoftPenalty] action={chosen_action} violated rule "
                    f"{rule.metric}({mv:.1f}){op}{th} → {self.penalty:.1f}"
                )

        return total_penalty


# ──────────────────────────────────────────────────────────────
# Ablation 2: Random Rule Agent
# LLM 호출 없이 매 에피소드 랜덤 룰을 생성한다.
# 동일한 sanitizer를 통과시켜 구조적 공정성을 확보한다.
# ──────────────────────────────────────────────────────────────

# forbidden_actions 샘플링 풀 (Action 9 제외 — sanitizer 통과와 일관성)
_ACTION_POOL = list(range(9))  # [0..8]

_METRIC_OPERATOR = ">="  # sanitizer가 <=, < 드롭 → 항상 >= 사용


class RandomRuleAgent(LLMStrategicAnalyst):
    """
    Ablation 2: LLM → 랜덤 룰 생성.
    동일한 SafetyRule 스키마, 동일한 sanitizer 통과,
    단 룰 내용(threshold, forbidden_actions)은 무작위 샘플링.

    목적: "LLM의 domain-aware 판단"이 없고 constraint 구조만 있을 때 어떤 결과가 나오는가?
    """

    def __init__(self, n_rules: int = 2):
        self.logger = Logger().get_logger()
        self.L2_strategy_applied_count = 0
        self.n_rules = n_rules
        # 시드 재설정 금지: 메인 스크립트에서 이미 설정된 global random state를 공유한다.
        # 여기서 다시 seed()를 호출하면 DRL epsilon-greedy 탐색 경로가 오염된다.
        self.logger.info(f"[RandomRuleAgent] Initialized. n_rules={n_rules}")

    def analyze_and_update_rules(
        self,
        episode_results: Dict[str, Any],
        previous_rules: list,
        episode_history: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:

        ep = episode_results.get("episode", "?")
        target_meet = episode_results.get("target_meet_rate", 100.0)

        # target_meet_rate < TARGET_CONSTRAINT → LLM 조건과 동일하게 빈 룰 반환
        if target_meet < TARGET_CONSTRAINT:
            xai_record = {
                "episode": ep, "reasoning": f"target_meet={target_meet:.1f}% < {TARGET_CONSTRAINT}%. No rules.",
                "critic_analysis": "Hard constraint.", "rules_generated": [],
                "n_rules": 0, "target_meet_rate": target_meet,
                "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
                "masked_actions": [],
            }
            return [], xai_record

        metrics = list(METRIC_STEP_RANGE.keys())
        chosen_metrics = random.sample(metrics, min(self.n_rules, len(metrics)))

        raw_rules = []
        for metric in chosen_metrics:
            lo, hi = METRIC_STEP_RANGE[metric]
            # threshold: uniform sample in range
            if isinstance(lo, int) and isinstance(hi, int):
                threshold = random.randint(int(lo), int(hi))
            else:
                threshold = round(random.uniform(float(lo), float(hi)), 1)

            # forbidden_actions: random subset of [0..8], size 1~5
            n_forbidden = random.randint(1, 5)
            forbidden = sorted(random.sample(_ACTION_POOL, n_forbidden))

            raw_rules.append(SafetyRule(
                metric=metric,
                operator=_METRIC_OPERATOR,
                threshold=threshold,
                forbidden_actions=forbidden,
            ))

        # 동일한 sanitizer 통과 (Action 9 보호, inverted 드롭, scale 클램프)
        rules = sanitize_rules(raw_rules, self.logger)

        reasoning = (
            f"[RANDOM] ep={ep} sampled metrics={[r.metric for r in rules]}, "
            f"thresholds={[r.threshold for r in rules]}, "
            f"forbidden={[r.forbidden_actions for r in rules]}"
        )
        self.logger.info(reasoning)

        xai_record = {
            "episode": ep,
            "reasoning": reasoning,
            "critic_analysis": "Random rule — no LLM reasoning.",
            "rules_generated": [r.model_dump() for r in rules],
            "n_rules": len(rules),
            "target_meet_rate": target_meet,
            "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
            "masked_actions": sorted(set(a for r in rules for a in r.forbidden_actions)),
        }

        if rules:
            self.L2_strategy_applied_count += 1

        return rules, xai_record


# ──────────────────────────────────────────────────────────────
# Ablation 4: Random Mask Agent
# train_ablation_randmask.py 에서 사용.
#
# Random Rule과의 차이:
#   Random Rule  — 조건(metric >= threshold)이 있고, 조건 충족 시에만 mask 발동
#   Random Mask  — 조건 없음. 에피소드마다 랜덤 액션 집합을 뽑아 매 스텝 무조건 차단
#
# 검증 목적: "LLM의 state-conditional masking이 중요한가,
#            아니면 어떤 식으로든 액션을 제한하면 효과가 있는가?"
#
# 인터페이스:
#   sample_episode_mask() → 에피소드 시작 시 호출, numpy bool mask 반환
#   get_episode_mask()    → 현재 에피소드 mask 반환 (훈련 루프에서 매 스텝 사용)
# ──────────────────────────────────────────────────────────────

# 에피소드당 차단할 액션 수 범위 [1, 4]
# LLM-HRL에서 룰 1~2개 × 평균 forbidden 2~3개 → 총 2~5개 수준에 맞춤
_RANDMASK_MIN_FORBIDDEN = 1
_RANDMASK_MAX_FORBIDDEN = 4


class RandomMaskAgent:
    """
    Ablation 4: 조건 없는 Random Mask.
    에피소드 시작마다 랜덤으로 N개 액션을 선택해 해당 에피소드 전체에서 무조건 차단.
    state_summary를 보지 않으므로 '언제 차단할지'에 대한 어떤 지능도 없다.

    LLM-HRL과의 비교:
      LLM-HRL     → KPI 관찰 → 룰 생성 → 상태 조건 평가 → 필요할 때만 mask
      Random Mask → 랜덤 선택 → 항상 mask (조건 없음)
    """

    def __init__(self, num_actions: int = 10):
        self.logger = Logger().get_logger()
        self.num_actions = num_actions
        self._current_mask = np.ones(num_actions, dtype=bool)  # 초기: 모든 액션 허용
        self.episode_mask_log = []   # XAI용: 에피소드별 차단 기록
        self.L2_strategy_applied_count = 0
        self.logger.info("[RandomMaskAgent] Initialized.")

    def sample_episode_mask(self, episode: int) -> np.ndarray:
        """
        에피소드 시작 시 호출. 이 에피소드에서 사용할 random mask를 새로 샘플링.
        Action 9는 항상 허용 (LLM-HRL의 sanitizer와 동일한 보호).
        Returns: bool array, True = 허용, False = 차단.
        """
        mask = np.ones(self.num_actions, dtype=bool)

        # [0..8] 중 랜덤하게 N개 차단 (Action 9는 항상 허용)
        n_forbidden = random.randint(_RANDMASK_MIN_FORBIDDEN, _RANDMASK_MAX_FORBIDDEN)
        forbidden = random.sample(_ACTION_POOL, n_forbidden)  # _ACTION_POOL = [0..8]

        for idx in forbidden:
            mask[idx] = False

        self._current_mask = mask
        self.L2_strategy_applied_count += 1

        log_entry = {"episode": episode, "forbidden_actions": sorted(forbidden)}
        self.episode_mask_log.append(log_entry)
        self.logger.info(f"[RandomMask] ep={episode+1} forbidden={sorted(forbidden)}")

        return mask

    def get_episode_mask(self) -> np.ndarray:
        """현재 에피소드의 mask 반환 (매 스텝 동일한 mask 적용)."""
        return self._current_mask
