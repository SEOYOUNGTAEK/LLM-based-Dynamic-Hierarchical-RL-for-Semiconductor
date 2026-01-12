import re
import numpy as np
from typing import List, Dict, Any, Literal, Annotated, Optional
from logger import Logger
from pydantic import BaseModel, Field



class SafetyRule(BaseModel):
    """
    L1에 적용될 안전 룰의 Pydantic 모델.
    'metric'을 'urgent_priority_lot_count'로만 제한합니다.
    """

    # ✅ (수정) 'metric'을 'urgent_priority_lot_count' 하나로만 제한
    metric: Literal["urgent_priority_lot_count"] = Field(...)

    operator: Literal[">=", ">", "<=", "<", "=="] = Field(...)
    threshold: int

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

    # 🚨 '_parse_state_summary' 함수를 아래 코드로 덮어쓰세요.
    def _parse_state_summary(self, state_summary: str) -> Dict[str, int]:
        """
        state_summary 텍스트를 파싱하여 'urgent_priority_lot_count'만 추출합니다.
        """
        try:
            # 시뮬레이션 텍스트에서 '긴급 물량 개수' 뒤에 적힌 숫자를 찾아 정수로 저장하되, 없으면 0으로 해라
            priority_match = re.search(r"urgent_priority_lot_count \(Priority=1, ID starts with U\): (\d+)", state_summary)
            priority_count = int(priority_match.group(1)) if priority_match else 0

            return {
                "urgent_priority_lot_count": priority_count  # ✅ 'priority'만 반환
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse state_summary: {e}. Summary: {state_summary[:50]}...")
            return {"urgent_priority_lot_count": 0}

    # 🚨 (수정) get_action_mask 함수 (더 간결하게)
    def get_action_mask(self, state_summary: str) -> np.ndarray:
        """
        Pydantic 룰('urgent_priority_lot_count' 기준)을 기반으로 마스크를 생성합니다.
        """
        mask = np.ones(self.num_actions, dtype=bool)
        state_metrics = self._parse_state_summary(state_summary)

        # 'urgent_priority_lot_count' 값 하나만 가져옴
        metric_value = state_metrics.get("urgent_priority_lot_count", 0)

        # L1 안전 계층(SimpleSafetyAgent)의 핵심 로직 : 상위 계층(L2)에서 전달받은 전략적 규칙을 바탕으로 강화학습 에이전트의 행동을 실시간으로 제한하는 액션 마스킹(Action Masking) 과정을 수행
        for rule in self.current_rules:
            try:
                # ✅상위 계층의 전략적 의도와 하위 계층의 실행 사이를 연결하는 스위치
                if rule.metric != "urgent_priority_lot_count":
                    continue

                is_triggered = False
                operator = rule.operator
                threshold = rule.threshold

                if (operator == "==" and metric_value == threshold):
                    is_triggered = True
                elif (operator == ">=" and metric_value >= threshold):
                    is_triggered = True
                elif (operator == ">" and metric_value > threshold):
                    is_triggered = True
                elif (operator == "<=" and metric_value <= threshold):
                    is_triggered = True
                elif (operator == "<" and metric_value < threshold):
                    is_triggered = True

                if is_triggered:
                    for forbidden_action_index in rule.forbidden_actions:
                        if 0 <= forbidden_action_index < self.num_actions:
                            mask[forbidden_action_index] = False
                            self.logger.warning(
                                f"[L1 Safety Mask] Rule triggered: "
                                f"{rule.metric} ({metric_value}) {operator} {threshold}. "
                                f"Action {forbidden_action_index} is MASKED."
                            )

            except Exception as e:
                self.logger.error(f"[L1 Safety Mask] Failed to process rule: {rule.model_dump_json()}", exc_info=True)

        return mask