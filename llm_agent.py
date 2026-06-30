# llm_agent.py
import json
from typing import List, Dict, Any, Optional, Literal, Union
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import numpy as np

from pydantic import BaseModel, Field
from utils import NumpyJSONEncoder
from logger import Logger
from openai import OpenAI


# ======================= 0. 전역 제약 상수 =======================
# 모든 조건(LLM/RuleBased)에서 공통 적용되는 target_meet_rate 하한선
# 실험 환경 상 90%는 사실상 달성 불가(Rule_FIFO 최고 avg 89.25%) → 85%로 현실화
TARGET_CONSTRAINT = 85.0

# ── 긴급 lot 강제 임계값 ────────────────────────────────────────
# [롤백] urgent>=1 강화는 역효과였다: 긴급 lot 4개 환경에서 "긴급이 1개라도
#   있으면 무조건 Action 9 강제"가 되어버려, DRL이 '긴급이 적을 때 다른 좋은
#   행동'을 배울 기회를 박탈 → greedy 정책이 왜곡(eval CT 5404 폭증).
# → threshold 2로 복귀. 긴급이 2개 이상 쌓였을 때만 Action 9 강제, 1개일 땐
#   DRL 자율. 단, 긴급 lot 수를 6개로 늘려(data_generator) 2개 동시 상황이
#   적당히 발생하도록 조정한다(개입 빈도와 학습 자율의 균형).
URGENT_FORCE_THRESHOLD = 2

# ── 메트릭별 step-level 평가 스케일 ──────────────────────────────
# L1(SimpleSafetyAgent)은 state_summary의 "스냅샷" 값으로 룰을 평가한다.
# 그런데 LLM은 episode_results의 "에피소드 누적" 값(예: iplt_over_count=483)을
# 보고 threshold를 잡기 때문에, 룰이 평가 스케일과 어긋나 영원히 발동되지 않는다.
# (실측: iplt 스냅샷 0~28, urgent 0~15, shortage 0 근처)
# → threshold를 이 범위로 클램프해 룰이 실제로 발동 가능하도록 보정한다.
METRIC_STEP_RANGE = {
    "urgent_priority_lot_count": (1, 8),      # 스냅샷 실측 0~15
    "iplt_over_lot_count":       (1, 25),     # 스냅샷 실측 0~28
    "production_shortage_pct":   (1.0, 20.0), # 시간보정 스냅샷
}


def sanitize_rules(rules, logger):
    """LLM이 생성한 룰을 코드레벨에서 안전하게 보정한다.

    실험 결과 발견된 3가지 LLM 룰 오류를 강제 교정:
    (1) Action 9 차단 — LLM이 'urgent<=2 → forbid[9]', 'shortage<=14 → forbid[9]' 등
        Action 9(긴급 lot 우선)를 잘못 차단 → DRL의 긴급 lot 학습 방해.
        Action 9는 어떤 경우에도 forbidden에서 제거한다.
    (2) 인버티드 로직 — 'higher=worse' 메트릭에 <=, < 연산자를 쓰면
        정상 상황(메트릭 낮음)에서 룰이 발동 → 항상 마스킹. 해당 룰 드롭.
    (3) 스케일 불일치 — episode 누적값 기준 threshold(예: iplt>=250)는
        step 스냅샷(0~28)에서 절대 도달 못 함 → step 범위로 클램프.
    """
    cleaned = []
    for r in rules:
        # (1) Action 9 보호
        if 9 in r.forbidden_actions:
            before = list(r.forbidden_actions)
            r.forbidden_actions = [a for a in r.forbidden_actions if a != 9]
            logger.warning(f"[L2-sanitize] Action 9 protected: {r.metric} {before} -> {r.forbidden_actions}")

        # (2) 인버티드 연산자 드롭 (3개 메트릭 모두 'high=문제'이므로 >=, > 만 유효)
        if r.operator in ("<=", "<"):
            logger.warning(f"[L2-sanitize] dropped inverted-logic rule: {r.metric} {r.operator} {r.threshold}")
            continue

        # (3) threshold step-range 클램프
        rng = METRIC_STEP_RANGE.get(r.metric)
        if rng and r.threshold > rng[1]:
            orig = r.threshold
            r.threshold = type(orig)(rng[1])
            logger.warning(f"[L2-sanitize] {r.metric} threshold {orig} -> {r.threshold} (step-max {rng[1]})")

        # (4) 긴급 룰 강화 — sparse 대응 (개입 강화)
        # 긴급 lot 룰의 threshold를 1로 낮춰, 긴급 lot이 1개라도 있으면 발동.
        # urgent 4개뿐이라 'urgent>=2'는 거의 안 터지므로 'urgent>=1'로 강제.
        if r.metric == "urgent_priority_lot_count" and r.threshold > URGENT_FORCE_THRESHOLD:
            orig = r.threshold
            r.threshold = type(orig)(URGENT_FORCE_THRESHOLD)
            logger.warning(f"[L2-sanitize] urgent threshold {orig} -> {r.threshold} (sparse intervention boost)")

        # (5) 빈 룰 제거
        if r.forbidden_actions:
            cleaned.append(r)
    return cleaned

# ======================= 1. Pydantic 스키마 정의 =======================

class SafetyRule(BaseModel):
    # Phase 2: 단일 메트릭 → 3개 메트릭으로 확장
    metric: Literal[
        "urgent_priority_lot_count",    # Process 5 버퍼의 긴급 Lot 수
        "production_shortage_pct",      # 생산 부족률 (%) = (target-output)/target*100
        "iplt_over_lot_count"           # IPLT 초과 Lot 수 (전체 라인)
    ] = Field(...)
    operator: Literal[">=", ">", "<=", "<", "=="] = Field(...)
    threshold: Union[float, int]        # float 허용 (shortage_pct는 소수점 가능)
    forbidden_actions: List[Annotated[int, Field(ge=0, le=9)]] = Field(
        description="차단할 액션 인덱스 리스트 (예: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
        default_factory=list
    )
    analysis: Optional[str] = None


class RuleResponse(BaseModel):
    reasoning: str = Field(
        description="High-level strategic analysis of the episode KPIs and historical trend."
    )
    critic_analysis: str = Field(
        description="Self-reflection on proposed rules: side-effects on target_meet_rate and IPLT."
    )
    rules: List[SafetyRule] = Field(
        description="Final validated safety rules for the L1 agent."
    )


# ======================= 2. LLM 분석가 클래스 =======================

class LLMStrategicAnalyst:

    def __init__(self):
        self.logger = Logger().get_logger()
        self.L2_strategy_applied_count = 0

        # --- Llama (Ollama) 클라이언트 설정 ---
        self.ollama_base_url = "http://localhost:11434/v1"
        self.ollama_model_name = "mistral"
        self.ollama_client = OpenAI(base_url=self.ollama_base_url, api_key='ollama')

        self.logger.info("L2 LLMStrategicAnalyst initialized with Ollama (Llama 3).")

    # --- Llama (Ollama) API 호출 헬퍼 ---
    def _call_llama(self, prompt: str) -> str:
        system_prompt = (
            "You are a display manufacturing process analyst expert in autonomous RL governance. "
            "You MUST respond with a single JSON object with keys: 'reasoning', 'critic_analysis', 'rules'. "
            "Each rule MUST include: 'metric', 'operator', 'threshold', 'forbidden_actions'. "
            "Allowed metrics: 'urgent_priority_lot_count', 'production_shortage_pct', 'iplt_over_lot_count'. "
            "threshold can be a float (e.g. 10.0 for shortage_pct). "
            # ── 핵심 제약: production_shortage_pct 룰은 Action 9를 절대 금지하지 않음 ──
            "CRITICAL: When metric is 'production_shortage_pct', Action 9 must NOT be in forbidden_actions. "
            "Use production_shortage_pct only to restrict IPLT-time actions [0,1,2], not Action 9. "
            "Action 9 may only be forbidden when urgent_priority_lot_count == 0 (no urgent lots at all). "
            "Correct example: {\"reasoning\": \"...\", \"critic_analysis\": \"...\", \"rules\": ["
            "{\"metric\": \"urgent_priority_lot_count\", \"operator\": \">=\", \"threshold\": 2, \"forbidden_actions\": [0,1,2,3,4,5,6,7,8]}, "
            "{\"metric\": \"production_shortage_pct\", \"operator\": \">=\", \"threshold\": 10.0, \"forbidden_actions\": [0,1,2]}]} "
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
    def _call_llm_for_rules(self, prompt: str) -> tuple:
        """
        Returns (rules, reasoning, critic_analysis).
        XAI를 위해 추론 과정도 함께 반환합니다.
        """
        try:
            json_string = self._call_llama(prompt)
            if not json_string:
                return [], "", ""

            raw_result = json.loads(json_string)
            validated_response = RuleResponse.model_validate(raw_result)

            self.logger.info(f"[L2 REASONING] {validated_response.reasoning}")
            self.logger.info(f"[L2 CRITIC] {validated_response.critic_analysis}")

            return validated_response.rules, validated_response.reasoning, validated_response.critic_analysis

        except Exception as e:
            self.logger.error(f"[L2] LLM call or validation failed: {e}")
            return [], "", ""

    # --- 전략 수립 및 성찰 메인 함수 (Phase 2: history + XAI 반환) ---
    def analyze_and_update_rules(
        self,
        episode_results: Dict[str, Any],
        previous_rules: List[SafetyRule],
        episode_history: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:  # (List[SafetyRule], xai_record: dict)

        self.logger.info(f"[L2] Analyzing episode results: {episode_results}")

        current_target_meet_rate = episode_results.get('target_meet_rate', 100.0)
        current_shortage_pct = max(0.0, 100.0 - current_target_meet_rate)

        # 히스토리 트렌드 요약 생성
        history_summary = self._summarize_history(episode_history or [])

        prompt = f"""
            You are an expert display manufacturing process manager governing a Hierarchical RL scheduler.
            Your goal is to create safety rules for the L1 Safety Agent to govern the RL agent (L0).

            [*** HARD CONSTRAINT — READ FIRST ***]
            Mandatory constraint: target_meet_rate MUST stay at or above {TARGET_CONSTRAINT}%.
            Current target_meet_rate = {current_target_meet_rate:.2f}%

            IF target_meet_rate < {TARGET_CONSTRAINT}:
              -> Return empty rules list [] IMMEDIATELY.
              -> This overrides ALL other goals.

            [*** CRITICAL: Threshold Scale ***]
            Rules are evaluated against PER-STEP SNAPSHOT values (not episode totals).
            Set thresholds within these observed per-step ranges, or the rule will NEVER fire:
              - urgent_priority_lot_count : 0 ~ 15  (usually 0-2)  → use threshold 2 (act when urgent lots accumulate)
              - iplt_over_lot_count       : 0 ~ 28  (usually 0-5)  → use threshold 5-20
              - production_shortage_pct   : 0 ~ 20  (usually 0-3)  → use threshold 5-15
            The episode summary below shows CUMULATIVE totals (much larger). Do NOT copy those
            numbers as thresholds. Use the per-step ranges above.
            Only operators >= and > are valid (rules trigger when a metric is HIGH = problematic).

            [***Available Metrics for Rules***]
            You may use ANY of these 3 metrics in your rules:

            1. urgent_priority_lot_count  (integer, per-step snapshot 0~15)
               - Number of urgent lots (ID starts with 'U') waiting in Process 5 buffer
               - Urgent lots are RARE (only 4 per episode) and easily neglected by DRL.
               - When urgent lots accumulate (>= 2) → FORCE Action 9 (forbid [0..8]); if only 1, let the agent decide

            2. production_shortage_pct  (float %, per-step snapshot 0~20)
               - Time-normalized: max(0, expected_output_by_now - actual_output) / target * 100
               - High value = production is falling behind → restrict IPLT-time actions [0,1,2], NOT Action 9

            3. iplt_over_lot_count  (integer, per-step snapshot 0~28)
               - Number of lots currently exceeding IPLT (quality degradation) threshold
               - High value = quality risk → use Actions 6-8 (forbid [0,1,2])

            [***Action Space***]
            - Actions 0-2: IPLT-time priority dispatch (restrict these when production is behind)
            - Actions 3-5: Production (Target Meet Rate) focused
            - Actions 6-8: IPLT management (quality protection)
            - Action 9: Urgent Lot speed priority
              *** Action 9 GUIDANCE ***
              - FORCE Action 9 (forbid [0~8]) when urgent_priority_lot_count >= 2
              - NEVER forbid Action 9 using production_shortage_pct as trigger
              - Action 9 may only be in forbidden_actions when urgent_priority_lot_count == 0

            [***Historical Trend (last {len(episode_history or [])} episodes)***]
            {history_summary}

            [***Current Episode Performance***]
            {json.dumps(episode_results, indent=2, cls=NumpyJSONEncoder)}

            [***Self-Reflection Chain-of-Thought***]
            (Skip and return [] if target_meet_rate < {TARGET_CONSTRAINT})

            1. TREND: Is performance improving, stable, or degrading vs. history?
            2. DIAGNOSE: Which metric is the root bottleneck this episode?
            3. PROPOSE: Write 1-2 rules using the most relevant metrics.
               CORRECT example:
               - IF urgent_priority_lot_count >= 2 → forbid [0,1,2,3,4,5,6,7,8]  (force Action 9 for urgent lots)
               - IF production_shortage_pct >= 10.0 → forbid [0,1,2]  (shift from IPLT-time to production focus)
               WRONG examples (never do these):
               - production_shortage_pct >= X → forbid [9]  ← FORBIDDEN
               - iplt_over_lot_count >= X → forbid [9]       ← FORBIDDEN unless urgent_lot_count == 0
            4. CRITIC: Will these rules cause target_meet_rate to drop below {TARGET_CONSTRAINT}%?
               Current margin = {current_target_meet_rate - TARGET_CONSTRAINT:.2f}%. If margin < 2%, do NOT add rule.
            5. REFINE: Output only safe, balanced rules.

            [***Output Format***]
            {{
              "reasoning": "Trend + diagnosis...",
              "critic_analysis": "Side-effect check with {TARGET_CONSTRAINT}% margin...",
              "rules": [...]
            }}
            """

        new_rules, reasoning, critic = self._call_llm_for_rules(prompt)

        # ── 코드레벨 룰 보정 (Action 9 보호 + 인버티드 드롭 + threshold 클램프) ──
        new_rules = sanitize_rules(new_rules, self.logger)

        # ── 코드레벨 Hard Constraint (LLM 프롬프트 우회 방지) ──
        # LLM이 제약을 무시하고 룰을 반환해도 코드에서 강제 클리어
        if current_target_meet_rate < TARGET_CONSTRAINT and new_rules:
            self.logger.warning(
                f"[L2] Code-level override: target_meet={current_target_meet_rate:.1f}% "
                f"< {TARGET_CONSTRAINT}%. LLM rules cleared (constraint enforcement)."
            )
            reasoning = f"[OVERRIDDEN] target_meet={current_target_meet_rate:.1f}% < {TARGET_CONSTRAINT}%. Rules cleared by code-level constraint."
            new_rules = []

        xai_record = {
            "episode":        episode_results.get("episode"),
            "reasoning":      reasoning,
            "critic_analysis": critic,
            "rules_generated": [r.model_dump() for r in new_rules],
            "n_rules":        len(new_rules),
            "target_meet_rate": episode_results.get("target_meet_rate"),
            "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
            "masked_actions": sorted(set(a for r in new_rules for a in r.forbidden_actions)),
        }

        if not new_rules:
            self.logger.warning("[L2] No new rules generated.")
            return [], xai_record

        self.L2_strategy_applied_count += 1
        return new_rules, xai_record

    def _summarize_history(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "No history available (first episodes)."

        lines = ["Episode | TargetMeet% | ULotCycleTime | ShortPct | IPLTOver"]
        for h in history:
            ep = h.get('episode', '?')
            tm = h.get('target_meet_rate', 0)
            ct = h.get('AVG_ULOT_CYCLE_TIME', 0)
            sp = max(0.0, 100.0 - tm)
            iplt = h.get('iplt_over_count', 0)
            lines.append(f"  Ep {ep:>3} | {tm:>10.1f} | {ct:>13.0f} | {sp:>8.1f} | {iplt:>8}")

        # 단순 추세: 마지막 vs 처음
        if len(history) >= 2:
            first_tm = history[0].get('target_meet_rate', 0)
            last_tm  = history[-1].get('target_meet_rate', 0)
            first_ct = history[0].get('AVG_ULOT_CYCLE_TIME', 0)
            last_ct  = history[-1].get('AVG_ULOT_CYCLE_TIME', 0)
            trend_tm = "improving" if last_tm > first_tm else ("stable" if abs(last_tm - first_tm) < 1 else "degrading")
            trend_ct = "improving" if last_ct < first_ct else ("stable" if abs(last_ct - first_ct) < 50 else "degrading")
            lines.append(f"  Trend: TargetMeet={trend_tm}, ULotCycleTime={trend_ct}")

        return "\n".join(lines)


# ======================= 조건 (B): 논문 원본 LLM 재현 =======================
# train_baseline_c.py 에서 사용
# 단일 메트릭(urgent_priority_lot_count), 90% 제약 없음, 히스토리 없음
# → 논문 결과의 "LLM-HRL Baseline (Hard Enforcement)" 원본 재현

class LLMAnalyst_OriginalPaper(LLMStrategicAnalyst):
    """
    조건 (B): 논문 원본 LLM-HRL 재현.
    단일 메트릭, 90% hard constraint 없음, 에피소드 히스토리 없음.
    """

    def analyze_and_update_rules(
        self,
        episode_results: Dict[str, Any],
        previous_rules: List[SafetyRule],
        episode_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[SafetyRule]:

        self.logger.info(f"[L2-B] (OriginalPaper) Analyzing: ep={episode_results.get('episode')}")

        priority_rate = episode_results.get('priority_process_rate', 100.0)
        avg_cycle_time = episode_results.get('AVG_ULOT_CYCLE_TIME', 0)
        target_meet = episode_results.get('target_meet_rate', 100.0)

        prompt = f"""
            You are an expert semiconductor process manager.
            Govern an RL agent (L0) by creating rules for a Safety Agent (L1).
            Prioritize balancing 'priority_process_rate' (Speed) and 'target_meet_rate' (Productivity).

            [***L1 Constraints***]
            - Only metric allowed: urgent_priority_lot_count
            - Action 3-5: Focus on Production (Target).
            - Action 9: Focus on Urgent Lots (Speed).

            [***Episode Performance Data***]
            {json.dumps(episode_results, indent=2, cls=NumpyJSONEncoder)}

            [***Strategic Judgment (Chain-of-Thought)***]

            1. (Failure: Under-Prioritization)
            IF priority_process_rate < 95% OR AVG_ULOT_CYCLE_TIME is high:
            -> Force Action 9.
            -> Rule: urgent_priority_lot_count >= 1, forbidden_actions: [0,1,2,3,4,5,6,7,8]

            2. (Failure: Overfitting)
            ELSE IF priority_process_rate > 90% AND target_meet_rate < 80%:
            -> Force Target actions.
            -> Rule: urgent_priority_lot_count == 0, forbidden_actions: [9]

            3. (Ideal: Balanced)
            ELSE:
            -> Remove all rules. Return [].

            Current values: priority_rate={priority_rate:.1f}%, target_meet={target_meet:.1f}%

            Respond ONLY with JSON: {{"reasoning": "...", "critic_analysis": "...", "rules": [...]}}
            """

        new_rules, reasoning, critic = self._call_llm_for_rules(prompt)
        # 코드레벨 룰 보정 (Action 9 보호 + 인버티드 드롭 + threshold 클램프)
        new_rules = sanitize_rules(new_rules, self.logger)
        xai_record = {
            "episode": episode_results.get("episode"),
            "reasoning": reasoning, "critic_analysis": critic,
            "rules_generated": [r.model_dump() for r in new_rules],
            "n_rules": len(new_rules),
            "target_meet_rate": episode_results.get("target_meet_rate"),
            "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
            "masked_actions": sorted(set(a for r in new_rules for a in r.forbidden_actions)),
        }
        if new_rules:
            self.L2_strategy_applied_count += 1
        return new_rules, xai_record


# ======================= 조건 (C): Phase 1 개선 LLM =======================
# train_baseline_b.py 에서 사용
# 단일 메트릭 + 90% hard constraint 추가, 히스토리 없음
# → Phase 1 기여(90% 제약)만 분리 검증

class LLMAnalyst_Phase1(LLMStrategicAnalyst):
    """
    조건 (C): Phase 1 개선 — 90% hard constraint 추가, 단일 메트릭.
    히스토리 없음. Phase 1 기여를 단독으로 측정.


    """

    def analyze_and_update_rules(
        self,
        episode_results: Dict[str, Any],

        previous_rules: List[SafetyRule],
        episode_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[SafetyRule]:

        self.logger.info(f"[L2-C] (Phase1) Analyzing: ep={episode_results.get('episode')}")

        current_target_meet_rate = episode_results.get('target_meet_rate', 100.0)
        priority_rate = episode_results.get('priority_process_rate', 100.0)

        # Hard Constraint (Phase 1 핵심 기여) — 기준: TARGET_CONSTRAINT
        if current_target_meet_rate < TARGET_CONSTRAINT:
            self.logger.warning(f"[L2-C] target_meet_rate={current_target_meet_rate:.1f}% < {TARGET_CONSTRAINT}%. Returning empty rules.")
            # ⚠️ 반드시 튜플 반환 (training script에서 new_rules, xai_rec = ... 으로 언패킹)
            xai_record = {
                "episode": episode_results.get("episode"),
                "reasoning": f"target_meet={current_target_meet_rate:.1f}% < {TARGET_CONSTRAINT}%. Hard constraint triggered.",
                "critic_analysis": "Rules cleared by Phase1 hard constraint.",
                "rules_generated": [], "n_rules": 0,
                "target_meet_rate": current_target_meet_rate,
                "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
                "masked_actions": [],
            }
            return [], xai_record

        prompt = f"""
            You are an expert display manufacturing process manager.
            Govern an RL agent (L0) via L1 Safety Agent rules.

            [*** HARD CONSTRAINT ***]
            target_meet_rate MUST stay >= {TARGET_CONSTRAINT}%. Current = {current_target_meet_rate:.2f}%.
            IF target_meet_rate < {TARGET_CONSTRAINT} -> return empty rules []. (Already checked above.)

            [***L1 Constraints***]
            - Only metric allowed: urgent_priority_lot_count
            - Action 9: Urgent lot speed priority
            - Safety margin to {TARGET_CONSTRAINT}% = {current_target_meet_rate - TARGET_CONSTRAINT:.2f}%. If < 2%, return [].

            [***Action 9 Rule Direction — READ CAREFULLY***]
            The ONLY valid rule pattern is:
              urgent_priority_lot_count >= N → forbidden_actions: [0,1,2,3,4,5,6,7,8]  (FORCE Action 9)
            Do NOT generate rules that include Action 9 in forbidden_actions.
            Forbidding Action 9 prevents DRL from learning urgent lot handling.

            [***Episode Performance***]
            {json.dumps(episode_results, indent=2, cls=NumpyJSONEncoder)}

            Current priority_process_rate = {priority_rate:.1f}%

            [***Chain-of-Thought***]
            1. DIAGNOSE: are urgent lots stuck in the buffer (high cycle time)?
            2. PROPOSE: if urgent lots are not being processed fast enough,
               force Action 9 with: urgent_priority_lot_count >= 2, forbidden_actions: [0,1,2,3,4,5,6,7,8]
            3. CRITIC: will forcing Action 9 drop target_meet_rate below {TARGET_CONSTRAINT}%?
               margin = {current_target_meet_rate - TARGET_CONSTRAINT:.2f}%. If < 2%, return [].
            4. REFINE: output only rules that FORCE Action 9, never block it.

            Respond ONLY with JSON: {{"reasoning": "...", "critic_analysis": "...", "rules": [...]}}
            """

        new_rules, reasoning, critic = self._call_llm_for_rules(prompt)
        # 코드레벨 룰 보정 (Action 9 보호 + 인버티드 드롭 + threshold 클램프)
        new_rules = sanitize_rules(new_rules, self.logger)
        xai_record = {
            "episode": episode_results.get("episode"),
            "reasoning": reasoning, "critic_analysis": critic,
            "rules_generated": [r.model_dump() for r in new_rules],
            "n_rules": len(new_rules),
            "target_meet_rate": episode_results.get("target_meet_rate"),
            "AVG_ULOT_CYCLE_TIME": episode_results.get("AVG_ULOT_CYCLE_TIME"),
            "masked_actions": sorted(set(a for r in new_rules for a in r.forbidden_actions)),
        }
        if new_rules:
            self.L2_strategy_applied_count += 1
        return new_rules, xai_record


# ======================= 조건 (E): Rule-based Mock LLM =======================
# train_baseline_rulebased.py 에서 사용
# LLM 호출 없이 동일한 if-else 규칙으로 대체
# → "LLM의 자연어 추론이 단순 규칙보다 실제로 나은가?" 검증

class LLMAnalyst_RuleBased(LLMStrategicAnalyst):
    """
    조건 (E): Rule-Based Mock LLM.
    Ollama 호출 없이 결정론적 if-else로 동일 목표를 달성.
    LLM 추론 품질 ablation: "LLM이 없어도 같은 결과가 나오는가?"
    """

    def __init__(self):
        self.logger = Logger().get_logger()
        self.L2_strategy_applied_count = 0
        self.logger.info("L2 RuleBased analyst initialized (no LLM call).")

    def analyze_and_update_rules(
        self,
        episode_results: Dict[str, Any],
        previous_rules: List[SafetyRule],
        episode_history: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:

        target_meet   = episode_results.get('target_meet_rate', 100.0)
        avg_ct        = episode_results.get('AVG_ULOT_CYCLE_TIME', 0)
        iplt_over     = episode_results.get('iplt_over_count', 0)

        # episode_history에서 평균 CT 계산 (기준선)
        history_ct_avg = 0.0
        if episode_history:
            ct_vals = [h.get('AVG_ULOT_CYCLE_TIME', 0) for h in episode_history if h.get('AVG_ULOT_CYCLE_TIME',0) > 0]
            if ct_vals:
                history_ct_avg = sum(ct_vals) / len(ct_vals)

        # ── 우선순위 1: 긴급 Lot CT가 히스토리 평균보다 10% 이상 나쁨 + 생산 OK ──
        # urgent_priority_lot_count 직접 사용 (priority_rate는 항상 100%)
        urgent_ct_bad = (avg_ct > history_ct_avg * 1.10) if history_ct_avg > 0 else (avg_ct > 4000)
        if urgent_ct_bad and target_meet >= TARGET_CONSTRAINT:
            rules = [SafetyRule(
                metric="urgent_priority_lot_count", operator=">=", threshold=2,
                forbidden_actions=[0,1,2,3,4,5,6,7,8]
            )]
            reasoning = (f"CT={avg_ct:.0f} > hist_avg*1.1={history_ct_avg*1.1:.0f}. "
                         f"Urgent lots not being handled fast enough. Force Action 9.")
            critic = f"target_meet={target_meet:.1f}% >= {TARGET_CONSTRAINT}%. Safe to enforce."

        # ── 우선순위 2: IPLT 품질 위험 ──
        # 발동 판단은 episode 누적값(iplt_over)으로, 룰 threshold는 step 스냅샷(0~28) 스케일로
        elif iplt_over > 100 and target_meet >= TARGET_CONSTRAINT:
            rules = [SafetyRule(
                metric="iplt_over_lot_count", operator=">", threshold=5,  # step 스냅샷 기준
                forbidden_actions=[0,1,2]   # IPLT-time 액션 제한 (Action 9는 유지)
            )]
            reasoning = (f"iplt_over={iplt_over} (episode total) > 100. Quality risk. "
                         f"Restrict IPLT-time actions [0,1,2] at step-level.")
            critic = f"target_meet={target_meet:.1f}% OK. IPLT is primary bottleneck."

        # ── 우선순위 3: 생산 부족 ── (TM 낮을 때 IPLT-time 액션 제한, Action 9 유지)
        elif target_meet < TARGET_CONSTRAINT:
            rules = [SafetyRule(
                metric="urgent_priority_lot_count", operator=">=", threshold=2,
                forbidden_actions=[0,1,2]   # IPLT-time 액션 제한, Action 9는 허용
            )]
            reasoning = (f"target_meet={target_meet:.1f}% < {TARGET_CONSTRAINT}%. "
                         f"Restrict IPLT-time dispatch to focus on production.")
            critic = "Removing IPLT-time priority to recover target_meet_rate."

        # ── 균형 상태 ──
        else:
            rules    = []
            reasoning = f"System balanced. TM={target_meet:.1f}%, CT={avg_ct:.0f}, IPLT={iplt_over}."
            critic    = "No intervention needed."

        # 안전망: 하드코딩 룰도 sanitize 통과 (Action 9 보호 + threshold 클램프)
        rules = sanitize_rules(rules, self.logger)

        xai_record = {
            "episode": episode_results.get("episode"),
            "reasoning": reasoning, "critic_analysis": critic,
            "rules_generated": [r.model_dump() for r in rules],
            "n_rules": len(rules),
            "target_meet_rate": target_meet,
            "AVG_ULOT_CYCLE_TIME": avg_ct,
            "masked_actions": sorted(set(a for r in rules for a in r.forbidden_actions)),
        }

        self.logger.info(f"[RuleBased] ep={episode_results.get('episode')} → {len(rules)} rules. {reasoning}")
        if rules:
            self.L2_strategy_applied_count += 1
        return rules, xai_record


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