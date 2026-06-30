import re
import numpy as np
from typing import List, Dict, Any, Literal, Annotated, Optional, Union
from logger import Logger
from pydantic import BaseModel, Field



class SafetyRule(BaseModel):
    """
    L1에 적용될 안전 룰의 Pydantic 모델.
    L2(llm_agent.py)의 SafetyRule과 동일한 3개 메트릭을 지원한다.
    (L1은 _parse_state_summary에서 3개 메트릭을 모두 파싱하므로,
     타입 정의도 L2와 일치시켜 정합성을 맞춘다.)
    """

    # ✅ (수정) L2와 동일하게 3개 메트릭 지원
    metric: Literal[
        "urgent_priority_lot_count",    # Process 5 버퍼의 긴급 Lot 수
        "production_shortage_pct",      # 생산 부족률 (시간 보정 %)
        "iplt_over_lot_count"           # IPLT 초과 Lot 수
    ] = Field(...)

    operator: Literal[">=", ">", "<=", "<", "=="] = Field(...)
    # ✅ (수정) shortage_pct는 소수점 가능 → L2와 동일하게 Union 허용
    threshold: Union[float, int]

    # ✅ (수정) 0~9까지 허용
    forbidden_actions: List[Annotated[int, Field(ge=0, le=9)]] = Field(
        description="The action indices to forbid (e.g., [3, 4, 5, 9]).", default_factory=list
    )
    analysis: Optional[str] = None


class SimpleSafetyAgent:
    def __init__(self, logger: Logger, num_actions: int):
        self.logger = logger
        self.current_rules: List[SafetyRule] = []
        self.num_actions = num_actions
        self.total_action_steps = 0  # 🚨 [추가] 총 스텝 카운터 (정책 스위칭 빈도 계산용)

        # 🚨 [추가] LLM 개입/추적 지표
        self.L2_strategy_applied_count = 0
        self.policy_switch_count = 0
        self.total_masked_actions = 0  # 실제 마스킹이 발생한 횟수

    def set_rules(self, new_rules: List[SafetyRule]):
        self.logger.info(f"[L1] Safety Rules UPDATED: {new_rules}")
        self.current_rules = new_rules
        if new_rules:
            self.L2_strategy_applied_count += 1

    def _parse_state_summary(self, state_summary: str) -> Dict[str, float]:
        """
        state_summary 텍스트를 파싱하여 3개 메트릭을 추출합니다.
        - urgent_priority_lot_count  (int)
        - production_shortage_pct    (float)
        - iplt_over_lot_count        (int)
        """
        try:
            priority_match = re.search(
                r"urgent_priority_lot_count.*?: (\d+)", state_summary)
            shortage_match = re.search(
                r"production_shortage_pct.*?: ([\d.]+)", state_summary)
            iplt_match = re.search(
                r"iplt_over_lot_count.*?: (\d+)", state_summary)

            return {
                "urgent_priority_lot_count": int(priority_match.group(1)) if priority_match else 0,
                "production_shortage_pct":   float(shortage_match.group(1)) if shortage_match else 0.0,
                "iplt_over_lot_count":       int(iplt_match.group(1)) if iplt_match else 0,
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse state_summary: {e}. Summary: {state_summary[:80]}...")
            return {"urgent_priority_lot_count": 0, "production_shortage_pct": 0.0, "iplt_over_lot_count": 0}

    def get_action_mask(self, state_summary: str) -> np.ndarray:
        """L2 룰 기반 Action Mask 생성."""
        mask, _ = self.get_action_mask_with_reason(state_summary)
        return mask

    def get_action_mask_with_reason(self, state_summary: str):
        """
        Action Mask + XAI용 트리거 이유를 함께 반환합니다.
        Returns: (mask: np.ndarray, triggered_rules: list[dict])
        triggered_rules 예시:
          [{"metric": "urgent_priority_lot_count", "value": 2,
            "operator": ">=", "threshold": 1,
            "forbidden_actions": [0,1,2,3,4,5,6,7,8],
            "reason": "urgent_priority_lot_count(2) >= 1"}]
        """
        mask = np.ones(self.num_actions, dtype=bool)
        state_metrics = self._parse_state_summary(state_summary)
        triggered_rules = []

        for rule in self.current_rules:
            try:
                metric_value = state_metrics.get(rule.metric, 0)
                op = rule.operator
                th = rule.threshold

                is_triggered = (
                    (op == "==" and metric_value == th) or
                    (op == ">=" and metric_value >= th) or
                    (op == ">"  and metric_value >  th) or
                    (op == "<=" and metric_value <= th) or
                    (op == "<"  and metric_value <  th)
                )

                if is_triggered:
                    for idx in rule.forbidden_actions:
                        if 0 <= idx < self.num_actions:
                            mask[idx] = False

                    triggered_rules.append({
                        "metric":          rule.metric,
                        "value":           metric_value,
                        "operator":        op,
                        "threshold":       th,
                        "forbidden_actions": rule.forbidden_actions,
                        "reason": f"{rule.metric}({metric_value:.2f}) {op} {th}",
                    })
                    self.logger.warning(
                        f"[L1] Rule triggered: {rule.metric}({metric_value:.2f}) {op} {th} "
                        f"→ forbid {rule.forbidden_actions}"
                    )

            except Exception as e:
                self.logger.error(f"[L1] Rule processing failed: {rule.model_dump_json()}", exc_info=True)

        return mask, triggered_rules