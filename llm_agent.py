# llm_agent.py
import json
from typing import List, Dict, Any, Optional, Literal, Annotated
import numpy as np

from pydantic import BaseModel, Field
from utils import NumpyJSONEncoder
from logger import Logger
from openai import OpenAI


# ======================= 1. Pydantic 스키마 정의 =======================

class SafetyRule(BaseModel):
    metric: Literal["urgent_priority_lot_count"] = Field(...)
    operator: Literal[">=", ">", "<=", "<", "=="] = Field(...)
    threshold: int
    forbidden_actions: List[Annotated[int, Field(ge=0, le=9)]] = Field(
        description="차단할 액션 인덱스 리스트 (예: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
        default_factory=list
    )
    analysis: Optional[str] = None


# llm_agent.py 내의 Pydantic 모델 수정
class RuleResponse(BaseModel):
    # 1. 현재 상황에 대한 진단 (Reasoning)
    reasoning: str = Field(
        description="Provide a high-level strategic analysis of the previous episode's KPIs. "
                    "Focus on the root causes of performance bottlenecks in the semiconductor FAB."
    )

    # 2. 제안된 규칙의 부작용 검토 (Self-Reflection / Critic)
    critic_analysis: str = Field(
        description="Perform a critical self-reflection on the proposed safety rules. "
                    "Analyze potential side effects, such as whether strict urgent lot prioritization "
                    "might cause a critical drop in overall target meet rate (productivity)."
    )

    # 3. 최종 확정된 규칙 리스트 (Finalized Rules)
    rules: List[SafetyRule] = Field(
        description="The final list of validated safety rules to be applied to the L1 agent, "
                    "refined through the self-reflection process."
    )


# ======================= 2. LLM 분석가 클래스 =======================

class LLMStrategicAnalyst:

    def __init__(self):
        self.logger = Logger().get_logger()
        self.L2_strategy_applied_count = 0

        # --- Llama (Ollama) 클라이언트 설정 ---
        self.ollama_base_url = "http://localhost:11434/v1"
        self.ollama_model_name = "llama3"
        self.ollama_client = OpenAI(base_url=self.ollama_base_url, api_key='ollama')

        self.logger.info("L2 LLMStrategicAnalyst initialized with Ollama (Llama 3).")

    # --- Llama (Ollama) API 호출 헬퍼 ---
    def _call_llama(self, prompt: str) -> str:
        system_prompt = (
            "You are a semiconductor process analyst expert in autonomous governance. "
            "You MUST respond with a single JSON object that strictly adheres to the requested format. "
            "The JSON object MUST contain 'reasoning', 'critic_analysis', and 'rules'. "
            "Each rule in 'rules' MUST include: 'metric', 'operator', 'threshold', and 'forbidden_actions'. "
            "Use ONLY the metric 'urgent_priority_lot_count'. "
            "Example: {\"reasoning\": \"...\", \"critic_analysis\": \"...\", \"rules\": [{\"metric\": \"urgent_priority_lot_count\", \"operator\": \">=\", \"threshold\": 1, \"forbidden_actions\": [0,1,2]}]} "
            "If no rules are needed, respond with an empty rules list."
        )

        response = self.ollama_client.chat.completions.create(
            model=self.ollama_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()

    # --- 메인 호출 로직 ---
    def _call_llm_for_rules(self, prompt: str) -> List[SafetyRule]:
        try:
            json_string = self._call_llama(prompt)
            if not json_string:
                return []

            raw_result = json.loads(json_string)
            validated_response = RuleResponse.model_validate(raw_result)

            # 성찰 내용 로깅 (XAI 및 논문 근거 활용)
            self.logger.info(f"[L2 REASONING] {validated_response.reasoning}")
            self.logger.info(f"[L2 CRITIC] {validated_response.critic_analysis}")

            return validated_response.rules

        except Exception as e:
            self.logger.error(f"[L2] LLM call or validation failed: {e}")
            return []

    # --- 전략 수립 및 성찰 메인 함수 ---
    def analyze_and_update_rules(self, episode_results: Dict[str, Any], previous_rules: List[SafetyRule]) -> List[
        SafetyRule]:

        self.logger.info(f"[L2] Analyzing episode results for Semiconductor Manufacturing: {episode_results}")

        prompt = f"""
            You are an expert semiconductor process manager.
            Your goal is to autonomously govern an RL agent (L0) by creating rules for a Safety Agent (L1).
            You MUST prioritize balancing 'priority_process_rate' (Speed) and 'target_meet_rate' (Productivity).

            [***L1 Constraints***]
            - Only metric allowed: `urgent_priority_lot_count`.
            - Action 3-5: Focus on Production (Target).
            - Action 9: Focus on Urgent Lots (Speed).

            [***Episode Performance Data***]
            {json.dumps(episode_results, indent=2, cls=NumpyJSONEncoder)}

            [***Self-Reflection Chain-of-Thought Process***]
            Follow these steps rigorously before finalizing your response:

            1. DIAGNOSE: Why did the agent fail or succeed in the last episode? 
               (e.g., Analyze cycle time of urgent lots vs. overall throughput.)

            2. PROPOSE: Suggest initial rules to fix the diagnosis. 
               (e.g., "If urgent lots are delayed, force Action 9.")

            3. CRITIC (The Self-Reflection Step): 
               - Evaluate the side effects of your PROPOSED rules.
               - "If I force Action 9, will the 'target_meet_rate' fall below the 90% target?"
               - "Is the explored state space too restricted by this rule?"
               - Current 'target_meet_rate' is {episode_results.get('target_meet_rate')}.

            4. REFINE: Adjust or discard the proposed rules based on the CRITIC analysis to ensure a robust balance.

            [***Output Format***]
            Respond ONLY with a JSON object:
            {{
              "reasoning": "Step 1 & 2 analysis...",
              "critic_analysis": "Step 3 critique and side-effect check...",
              "rules": [...]
            }}
            """

        new_rules = self._call_llm_for_rules(prompt)

        if not new_rules:
            self.logger.warning("[L2] No new rules generated. Returning empty list.")
            return []

        self.L2_strategy_applied_count += 1
        return new_rules



'''
1. 현재 프롬프트 (Baseline):

Python

# llm_agent.py 내의 analyze_and_update_rules 함수
prompt = f"""
        You are an expert semiconductor process manager.
        ... (생략: 지시사항 및 Action 정의) ...

        [***Strategic Judgment (Chain-of-Thought)***]

        **1. (Failure Mode: Under-Prioritization)**
        IF 'priority_process_rate' is too low (< 95%) 
          OR ('AVG_ULOT_CYCLE_TIME' is greater than historical best):
        -> **Diagnosis:** L0 is not prioritizing speed/optimal path for urgent lots...
        -> **Action:** Force L0 to learn Action 9.
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": ">=", "threshold": 1, "forbidden_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]}} # ⬅️ 수정
        
        **2. (Failure Mode: Overfitting)**
        ELSE IF 'priority_process_rate' is high (> 90%) AND 'target_meet_rate' is too low (< 80%): 
        -> **Diagnosis:** L0(DQN) is overfitting on Action 9...
        -> **Action:** Force L0 to learn Target-focused actions.
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": "==", "threshold": 0, "forbidden_actions": [9]}} # ⬅️ 수정

        **3. (Ideal State: Balanced)**
        ELSE:
        -> **Diagnosis:** The system is balanced. L0 needs autonomy.
        -> **Action:** Remove all rules.
        -> **Rule Generation:** Return an empty list `[]`.

        ... (생략: Output Format) ...
        """
        
2. Variation 1: Soft Enforcement (비용/효율 기반 허용)
LLM이 속도 강제(Action 9)를 지시하기 전에, 생산량(target_meet_rate)이 85% 미만으로 떨어질 경우, 속도 강제 대신 생산성을 우선하도록 전략적 트레이드오프 판단을 추가합니다.

재수정 >> Target Meet Rate 임계값을 **85%에서 90%**로 상향 조정합니다.

        [***Strategic Judgment (Chain-of-Thought)***]

        **1. (Failure Mode: Under-Prioritization) - Soft Enforcement**
        IF (
              ('priority_process_rate' is too low (< 95%) OR ('AVG_ULOT_CYCLE_TIME' is greater than historical best))
              AND 
              ('target_meet_rate' is acceptable (>= 90%)) # ⬅️ [수정]: 임계값을 85% -> 90%로 상향
           ):
        -> **Diagnosis:** L0 is not prioritizing speed/optimal path, BUT target meet rate is acceptable. MUST force Action 9.
        -> **Action:** Force L0 to learn Action 9.
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": ">=", "threshold": 1, "forbidden_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]}} 

        **2. (Failure Mode: Critical Target Failure) - Prioritize Production**
        ELSE IF 'target_meet_rate' is critically low (< 90%): # ⬅️ [수정]: 임계값을 85% -> 90%로 상향
        -> **Diagnosis:** Production target is critically low. Temporarily suspend speed enforcement to boost output.
        -> **Action:** Remove all rules. The immediate priority is boosting production.
        -> **Rule Generation:** Return an empty list `[]`.

        **3. (Failure Mode: Overfitting)**
        ELSE IF 'priority_process_rate' is high (> 90%) AND 'target_meet_rate' is too low (< 80%):
        -> **Diagnosis:** L0(DQN) is overfitting on Action 9...
        -> **Action:** Force L0 to learn Target-focused actions.
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": "==", "threshold": 0, "forbidden_actions": [9]}} 

        **4. (Ideal State: Balanced)**
        ELSE:
        -> **Diagnosis:** The system is balanced. L0 needs autonomy.
        -> **Action:** Remove all rules.
        -> **Rule Generation:** Return an empty list `[]`.

        ... (생략: Output Format) ...
        """
        
3. Variation 2: Policy Exploration (불확실성 관리)
LLM이 최적의 정책에 도달하여 시스템이 **균형 상태(안정)**에 있을 때, 현재 정책을 강제하는 Action 9를 금지하고 다른 Action(Action 0-8)을 탐색하도록 강제하여 더 나은 정책이 있는지 검증하도록 유도합니다.

프롬프트 수정 사항 (Variation 2):

Python

# llm_agent.py 내의 analyze_and_update_rules 함수
prompt = f"""
        You are an expert semiconductor process manager.
        ... (생략: 지시사항 및 Action 정의) ...

        [***Strategic Judgment (Chain-of-Thought)***]

        **1. (Failure Mode: Under-Prioritization) - Hard Enforcement (유지)**
        IF 'priority_process_rate' is too low (< 95%) 
          OR ('AVG_ULOT_CYCLE_TIME' is greater than historical best):
        -> **Diagnosis:** L0 is not prioritizing speed/optimal path for urgent lots...
        -> **Action:** Force L0 to learn Action 9.
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": ">=", "threshold": 1, "forbidden_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]}} 

        **2. (Failure Mode: Overfitting) - Exploration Required** # ⬅️ [신규/수정 모드] 균형 상태에서 탐색 강제
        ELSE IF 'priority_process_rate' >= 99% 
                AND 'AVG_ULOT_CYCLE_TIME' is at historical best (or low) 
                AND 'POLICY_SWITCH_COUNT' is very low (< 5): # ⬅️ [신규 조건] 정책 안정성 지표 활용
        -> **Diagnosis:** Policy has converged and is highly stable, but exploration is needed to find potentially better long-term policies.
        -> **Action:** Temporarily force Action 9 to be forbidden to drive DRL to explore production-focused policies (Action 0-8).
        -> **Rule Generation:** Generate a rule: {{"metric": "urgent_priority_lot_count", "operator": "==", "threshold": 0, "forbidden_actions": [9]}} 
        # (Action 9를 금지하여 Target/IPLT 기반 Action(0-8) 탐색 유도)
        
        **3. (Default State: Balanced or Other)** # ⬅️ Ideal State 로직을 Default로 이동
        ELSE:
        -> **Diagnosis:** The system is performing adequately or is in an unstable state not covered by the primary modes. L0 needs full autonomy for now.
        -> **Action:** Remove all rules.
        -> **Rule Generation:** Return an empty list `[]`.

        ... (생략: Output Format) ...
        """
'''