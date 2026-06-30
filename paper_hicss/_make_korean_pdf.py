# -*- coding: utf-8 -*-
"""
한국어 HICSS 논문 PDF 생성기
Malgun Gothic (Windows 기본 한국어 폰트) 사용
"""
from fpdf import FPDF
import re, os

FONT_REG  = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"
OUT_PATH  = r"E:\SYT\SID_1031\SID_LLM_1031\iplt\paper_hicss\hicss_paper_ko.pdf"

# ── 텍스트 정제 함수 ─────────────────────────────────────────────────────
def clean(t):
    t = re.sub(r'\\cite\{[^}]*\}', '', t)
    t = re.sub(r'\\ref\{[^}]*\}', '', t)
    t = re.sub(r'\\label\{[^}]*\}', '', t)
    t = re.sub(r'\\textbf\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\emph\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\textit\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\texttt\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\noindent\b', '', t)
    t = re.sub(r'\\smallskip\b', '', t)
    t = re.sub(r'\\setlength[^}]*\}[^}]*\}', '', t)
    t = re.sub(r'\\\w+\*?\{[^}]*\}', '', t)
    t = re.sub(r'\$[^$]*\$', lambda m: math_to_text(m.group()), t)
    t = t.replace('\\%', '%').replace('\\,', ' ').replace('\\;', ' ')
    t = t.replace('---', '—').replace('--', '–').replace('``', '"').replace("''", '"')
    t = t.replace('{,}', ',').replace('\\$', '$')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def math_to_text(m):
    m = m.strip('$')
    m = m.replace(r'\hat{\rho}', 'rho^').replace(r'\rho', 'rho').replace(r'\tau', 'τ')
    m = m.replace(r'\Delta\rho', 'Δrho').replace(r'\gamma', 'γ').replace(r'\theta', 'θ')
    m = m.replace(r'\mathcal{S}', '𝒮').replace(r'\mathcal{A}', 'S')
    m = m.replace(r'\overline{\mathrm{CT}}_i', 'CT_i').replace(r'\times', '×')
    m = m.replace(r'\geq', '>=').replace(r'\leq', '<=').replace(r'\approx', '~=')
    m = m.replace(r'\in', 'in').replace(r'\setminus', chr(92)).replace(r'\cup', 'U')
    m = m.replace(r'\bigcup', 'U').replace(r'\emptyset', '{}').replace(r'\arg\max', 'argmax')
    m = m.replace(r'\mathrm', '').replace(r'\text', '').replace(r'\mathbf', '')
    m = re.sub(r'[{}\\]', '', m)
    return m.strip()

# ── PDF 클래스 ────────────────────────────────────────────────────────────
class KoreanPaper(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.add_font('Malgun',  '', FONT_REG)
        self.add_font('Malgun',  'B', FONT_BOLD)
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(True, margin=20)
        self._sec_num = 0
        self._subsec_num = 0

    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font('Malgun', '', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f'- {self.page_no()} -', align='C')
        self.set_text_color(0, 0, 0)

    # ── 스타일 메서드 ────────────────────────────────────────────────────
    def title_block(self, title, subtitle=None):
        self.set_font('Malgun', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 9, title, align='C')
        if subtitle:
            self.set_font('Malgun', 'B', 14)
            self.multi_cell(0, 8, subtitle, align='C')
        self.ln(4)

    def abstract_block(self, text):
        self.set_font('Malgun', 'B', 10)
        self.cell(0, 6, '초록', ln=True)
        self.set_font('Malgun', '', 9)
        x = self.get_x()
        self.set_left_margin(25)
        self.set_right_margin(25)
        self.multi_cell(0, 5, clean(text))
        self.set_left_margin(20)
        self.set_right_margin(20)
        self.ln(3)

    def keywords(self, kw):
        self.set_font('Malgun', 'B', 9)
        self.cell(18, 5, '키워드: ', ln=False)
        self.set_font('Malgun', '', 9)
        self.multi_cell(0, 5, clean(kw))
        self.ln(2)

    def section(self, num, title):
        self.ln(4)
        self.set_font('Malgun', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 7, f'{num}. {clean(title)}', ln=True, fill=True)
        self.ln(2)

    def subsection(self, title, star=False):
        self.ln(3)
        self.set_font('Malgun', 'B', 10.5)
        label = clean(title)
        self.multi_cell(0, 6, label)
        self.ln(1)

    def body(self, text):
        self.set_font('Malgun', '', 9.5)
        t = clean(text)
        if t:
            self.multi_cell(0, 5.5, t)
            self.ln(1.5)

    def bullet(self, label, text):
        self.set_font('Malgun', 'B', 9.5)
        lw = self.get_string_width(label + '  ')
        self.cell(lw, 5.5, label + '  ', ln=False)
        self.set_font('Malgun', '', 9.5)
        x0 = self.get_x()
        y0 = self.get_y()
        self.multi_cell(0, 5.5, clean(text))
        self.ln(1)

    def equation(self, label, eq_text):
        self.ln(2)
        self.set_font('Malgun', '', 9)
        self.set_text_color(40, 40, 120)
        self.multi_cell(0, 5, f'  [{label}]  {eq_text}', align='C')
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def figure_placeholder(self, label, caption):
        self.ln(2)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(150, 150, 150)
        self.rect(self.get_x(), self.get_y(), 170, 18, 'DF')
        self.set_font('Malgun', '', 8.5)
        self.set_text_color(80, 80, 80)
        self.set_xy(self.get_x(), self.get_y() - 18)
        self.cell(170, 18, f'[그림: {label}]', align='C', ln=True)
        self.set_draw_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        self.set_font('Malgun', '', 8.5)
        self.multi_cell(0, 5, '그림 설명: ' + clean(caption))
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def table_2col(self, caption, headers, rows):
        self.ln(2)
        self.set_font('Malgun', 'B', 8.5)
        self.multi_cell(0, 5, '표: ' + clean(caption))
        self.ln(1)
        cw = [60, 25, 25, 25, 25, 20]
        used = sum(cw[:len(headers)])
        if used > 170:
            factor = 170 / used
            cw = [c * factor for c in cw]
        # header
        self.set_fill_color(210, 220, 240)
        self.set_font('Malgun', 'B', 8)
        for i, h in enumerate(headers):
            w = cw[i] if i < len(cw) else 20
            self.cell(w, 6, clean(h), border=1, fill=True, align='C')
        self.ln()
        # rows
        self.set_fill_color(255, 255, 255)
        self.set_font('Malgun', '', 8)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            self.set_fill_color(248, 248, 248 if fill else 255)
            for i, cell in enumerate(row):
                w = cw[i] if i < len(cw) else 20
                txt = clean(str(cell))
                self.cell(w, 6, txt, border=1, fill=fill, align='C')
            self.ln()
        self.set_fill_color(255, 255, 255)
        self.ln(2)


# ─────────────────────────────────────────────────────────────────────────
#  논문 내용 빌드
# ─────────────────────────────────────────────────────────────────────────
pdf = KoreanPaper()
pdf.add_page()

# ── 제목 ─────────────────────────────────────────────────────────────────
pdf.title_block(
    '비결정적 디지털 트윈에서의 신뢰 가능한 스케줄링을 위한 LLM 거버넌스 기반 심층강화학습'
)

# ── 초록 ─────────────────────────────────────────────────────────────────
abstract = """
심층강화학습(DRL)은 제조 스케줄링 분야에 점점 더 많이 적용되고 있으나,
학습된 정책의 신뢰성 문제가 존재한다. 고충실도 디지털 트윈의 내재적 비결정성으로
인해, 동일하게 설정된 학습 실행이 어떤 경우에는 우수한 정책으로 수렴하고
어떤 경우에는 치명적으로 실패하기도 한다.
본 연구는 반도체 제조 디지털 트윈에서의 긴급 로트 스케줄링을 분석하여,
긴급 로트가 스케줄링 충돌을 야기할 만큼 충분히 빈번한 경우
DRL이 독립 실행의 60%에서 치명적으로 실패함을 보인다.
본 논문은 로컬 호스팅 대형언어모델(LLM)이 에피소드 단위 거버넌스 계층으로서
DRL 에이전트를 제약하는 안전 룰을 생성하는 3계층 계층적 강화학습(HRL) 프레임워크를 제안한다.
5개의 독립 시드에 걸쳐, Pure DRL은 60%의 실행에서 실패한 반면
LLM 거버넌스 에이전트는 0% 실패율을 달성하여, 생산 목표 달성률의 손실 없이
사이클 타임 분산을 73% 감소시켰다.
LLM 거버넌스를 내결함성 메커니즘으로 정의하여, 사회기술적 제조 시스템에서
학습 컴포넌트의 신뢰성을 향상시킴을 논한다.
"""
pdf.abstract_block(abstract)
pdf.keywords('신뢰성 AI; 심층강화학습; 대형언어모델; 디지털 트윈; 제조 스케줄링')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(1, '서론')
pdf.body("""
반도체 제조 라인은 확률적이고 비정상적인 스케줄링 환경이다. 단 하나의 디스패칭 결정이
공정 내 재공품(WIP) 분포를 재편하고, 수십 분에서 수 시간 후의 사이클 타임에 영향을
미칠 수 있다. 따라서 시점 t에서 스케줄러의 행동은 고정 휴리스틱으로는 귀속하기도,
최적화하기도 어려운 장기적 결과를 수반한다. 이러한 복잡성에 대처하기 위해 최근 연구들은
라인의 고충실도 이산 이벤트 시뮬레이션인 디지털 트윈과 심층강화학습(DRL)을 결합하여
시뮬레이션 환경에서 직접 디스패칭 정책을 학습하고 있다.
""")
pdf.figure_placeholder('DRL_Framework', """기본 DRL 스케줄링 프레임워크. 디지털 트윈이 학습 환경으로 작동하며,
에이전트는 공장 현장 상태 s를 관찰하고 디스패칭 행동 a를 선택한 뒤 Q-네트워크 가중치를 갱신한다.""")
pdf.body("""
고무적인 평균 성능에도 불구하고, 비용·수율이 중요한 제조에 DRL을 배포하는 것은
문헌에서 좀처럼 정량화되지 않는 신뢰성(dependability) 문제를 야기한다: 학습된 정책이
실행마다 신뢰할 수 없다. 본 논문에서는 디지털 트윈의 내재적 비결정성으로 인해,
동일하게 설정된 DRL 학습 실행이 어떤 실행에서는 고품질 정책으로 수렴하고 다른
실행에서는 치명적으로 실패할 수 있음을 증명한다.
이러한 취약성은 긴급 로트 충돌이 발생할 때 가장 심각하다: 고우선순위 소규모 로트
집단이 가능한 한 빠르게 라인을 통과해야 하지만, 병목 단계에서 진짜 경합을 일으킬
만큼 많아지면 보상 신호가 노이즈가 많아지고 에이전트의 수렴이 우연의 문제가 된다.
""")
pdf.body("""
이러한 실행 간 비신뢰성은 신뢰성 공학이 특성화하고 관리하려는 결함 유형과 정확히
일치한다. 고전적 분류 체계(Laprie, 2004)는 신뢰성을 속성(신뢰성, 가용성), 위협
(결함·오류·장애), 수단(결함 예방·허용·제거·예측)으로 구성한다. 따라서 우리가 제기하는
공학적 질문은 "DRL을 평균적으로 어떻게 더 빠르게 만드는가?"가 아니라 "어떻게
신뢰할 수 없는 DRL 스케줄러를 신뢰할 수 있게 만드는가?"이다.
""")
pdf.body("""
우리는 이 질문에 LLM이 전술적 DRL 에이전트에 대한 전략적 거버넌스를 제공하는
3계층 계층적 강화학습(HRL) 프레임워크로 답한다. 각 학습 에피소드 후, LLM은 집계된
KPI를 검사하고 소수의 안전 룰을 생성하며, 이 룰들은 DRL 에이전트를 제약하는
스텝별 행동 마스크로 컴파일된다. 결정적으로 LLM은 외부 API 호출 없이 로컬로 호스팅된다.
""")
pdf.subsection('본 논문의 기여 (세 가지)')
pdf.bullet('기여 1:', '디지털 트윈 비결정성 하에서의 DRL 비신뢰성 실증적 특성화. 동일한 설정이 임계 희소성 체제에서 독립 실행의 60%에서 실패함을 보인다.')
pdf.bullet('기여 2:', '내결함성 계층으로서의 LLM 거버넌스. 에피소드 단위 LLM 생성 룰이 치명적 실패율을 60%에서 0%로 줄이고 사이클 타임 분산을 73% 감소시킨다.')
pdf.bullet('기여 3:', '로컬 LLM 룰 생성을 위한 배포 교훈. 7B 로컬 모델의 체계적인 룰 생성 오류 패턴과 배포를 안전하게 만든 가드레일을 보고한다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(2, '관련 연구')
pdf.subsection('관련 접근법 비교')
pdf.body('아래 표는 본 프레임워크를 산업 배포에 중요한 다섯 가지 차원에서 관련 연구들과 비교한다.')
pdf.table_2col(
    '다섯 가지 배포 관련 차원에서의 LLM 안내 RL 접근법 비교 (빈도/안전/로컬/해석/신뢰성)',
    ['접근법', '빈도', '안전', '로컬', '해석', '신뢰성'],
    [
        ['Du (2023)',         '단발성',    'X', 'X', 'X', 'X'],
        ['Carta (2023)',      '매 스텝',   'X', 'X', 'X', 'X'],
        ['Liang (2023)',      '단발성',    '부분','X','부분','X'],
        ['Hasanbeig (2020)',  '정적',      'O', 'X', 'X', 'X'],
        ['본 연구 (LLM-HRL)', '에피소드',  'O', 'O', 'O', 'O'],
    ]
)
pdf.body('핵심 차별점은 지속적·KPI 기반 개정 주기, 산업 데이터 보안 요건을 충족하는 로컬 모델 사용, 그리고 신뢰성 기준(실패율 감소)에 대한 명시적 검증이다.')

pdf.subsection('스케줄링을 위한 DRL 및 디지털 트윈')
pdf.body('강화학습은 반도체 제조 디스패칭 및 스케줄링에 정적 규칙 기반 정책의 대안으로 적용되어 왔다. 전술 계층으로 채택한 Rainbow DQN 아키텍처(Hessel et al., 2018)는 이중 Q학습, 듀얼링 네트워크, 우선순위 재플레이, 노이즈 탐색, 다단계 리턴을 결합하며 이산 행동 제어의 강력한 기본값이다. 그러나 이 문헌은 소수의 실행에 걸친 평균 성능을 주로 보고하며 신뢰성(독립 학습 전반의 결과 분포)을 거의 분석하지 않는다.')

pdf.subsection('희소 보상과 탐색')
pdf.body('DRL의 잘 알려진 장애물은 희소 보상 문제이다: 보상받아야 할 사건이 드물 때 에이전트는 신호를 거의 받지 못하고 이를 무시하는 지역 최적해에 안착할 수 있다. 본 연구에서는 보상을 재형성하는 대신 외부 거버너를 통해 행동 공간을 제약하여, 보상 함수를 변경하지 않고 에이전트를 원하는 행동으로 유도한다.')

pdf.subsection('강화학습을 안내하는 LLM')
pdf.body('LLM은 최근 RL에 고수준 지식을 주입하는 데 사용되어 왔다. 본 프레임워크는 두 가지 면에서 차별화된다. 첫째, 거버넌스는 지속적이고 KPI 기반이다: LLM이 매 에피소드 라인 상태를 재평가하고 룰을 개정한다. 둘째, 거버넌스는 빈도에 의해 분리된다: LLM은 에피소드 시간 척도에서 작동하고, 경량 안전 에이전트가 밀리초 디스패치 시간 척도에서 결과 룰을 집행한다.')

pdf.subsection('AI/ML 컴포넌트의 신뢰성')
pdf.body('신뢰성 공학은 정확성이 보장되지 않는 시스템에 대한 추론 어휘를 제공한다. 학습 컴포넌트는 고전 소프트웨어에 없는 결함 모드를 도입한다: 분포 이동, 불투명한 결정 경계, 그리고 본 논문의 핵심인 비결정적 학습 결과. 본 연구는 DRL 스케줄러를 간헐적 결함을 가진 AI 컴포넌트로 취급하고, 그 실패율을 분포적으로 측정하며, LLM 거버넌스를 명시적인 내결함성 메커니즘으로 도입한다.')

pdf.subsection('결함—오류—장애 매핑 (Laprie 분류체계 적용)')
pdf.bullet('결함(Fault):', '디지털 트윈 내의 확률적 시뮬레이터 수준 비결정성 — 스케줄러가 관찰하거나 제어할 수 없는 내재적 요소.')
pdf.bullet('오류(Error):', '긴급 로트를 과소우선시하는 학습된 DRL 정책 — 결함이 학습 중 보상 신호를 오염시킬 때 생성되는 잠재적 중간 상태.')
pdf.bullet('장애(Failure):', '평균 CT_i > τ = 3,500s — 공장 현장에서 수율 및 납기 패널티를 유발하는 관측 가능한 서비스 수준 결과.')
pdf.body('따라서 LLM 거버넌스는 내결함성 메커니즘으로 위치한다: 결함(비결정성)을 제거하는 것이 아니라 결함 → 오류 → 장애 전파 사슬을 차단한다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(3, '3계층 거버넌스 아키텍처')
pdf.figure_placeholder('fig2_architecture_improved', """3계층 거버넌스 아키텍처. L2(Mistral-7B)는 KPI로부터 에피소드 단위 안전 룰을 생성하고,
L1(안전 에이전트)은 룰을 검증·집행하며, L0(Rainbow DQN)는 허용된 행동 중 최고 가치 행동을 선택한다.""")
pdf.body('본 프레임워크는 서로 다른 시간 척도에서 운영되는 세 계층에 걸쳐 전략과 전술을 분리한다. 룰 새니타이저는 별도 계층이 아닌 L1 내에 전처리 단계로 내장되어 있다.')

pdf.subsection('L0 — 전술 에이전트 (Rainbow DQN)')
pdf.body('최하위 계층은 21차원 상태 벡터를 관찰하고 매 결정 스텝에서 10개의 디스패칭 행동 중 하나를 선택하는 Rainbow DQN 에이전트이다. 상태는 정규화된 시뮬레이션 시간(1), 디바이스 패밀리별 공정 단계 버퍼 점유율(4), 디바이스별 생산 부족(4), 공정 내 리드타임 위험 통계(8), 임계 공정 단계에서 대기 중인 긴급 로트 수(4)를 연결한다.')
pdf.equation('보상 함수', 'R(s,a) = R_prod(s,a) + R_urgent(s,a) + R_penalty(s,a)')
pdf.body('R_prod는 생산 달성률, R_urgent는 긴급 로트 사이클 타임 감소 보상, R_penalty는 우선순위 위반 페널티를 포착한다. 긴급 로트가 전체 로트의 최대 7.7%를 차지하여 R_urgent가 간헐적이고 희소 신호 조건을 형성한다.')

pdf.subsection('L1 — 안전 집행 (내장 새니타이저 포함)')
pdf.body('중간 계층은 L2가 생성한 룰 집합 R = {r1, ..., rK}을 집행하는 경량 안전 에이전트이다. 집행 전에 내장 룰 새니타이저가 각 입력 룰을 검증하여, 논리적으로 역전된 조건자를 폐기하고, 행동 9의 억제를 방지하며, 임계값을 스텝별 평가 척도로 클램핑한다.')
pdf.body('각 검증된 룰 r_k는 튜플 (m_k, ⊙_k, θ_k, F_k)이며, m_k는 KPI 지표, ⊙_k는 비교 연산자, θ_k는 임계값, F_k는 금지된 행동 집합이다.')
pdf.equation('안전 행동 집합', 'A_safe(s) = S \ U_{k: m_k(s)⊙_k θ_k} F_k')
pdf.equation('최적 행동 선택', 'a* = argmax_{a in A_safe(s)} Q(s, a; θ)')
pdf.body('A_safe(s) = {}인 경우(충돌하는 룰이 모든 행동 금지), 폴백이 제약 없는 탐욕적 행동으로 되돌아가 데드락을 방지한다.')

pdf.subsection('L2 — 전략적 거버넌스 (로컬 LLM)')
pdf.body('최상위 계층은 에피소드 단위 거버넌스를 수행하는 로컬 호스팅 LLM이다. 각 학습 에피소드 후 집계된 KPI(긴급 로트 수, 생산 부족 비율, IPLT 초과 로트 수)와 5에피소드 롤링 히스토리를 받아, (i) 성능 추세 분석, (ii) 지배적 병목 진단, (iii) 안전 룰 생성을 수행한다. LLM은 자연어로 추론하고 구조화된(JSON) 응답을 반환하여 모든 룰에 대한 인간이 읽을 수 있는 근거를 제공한다.')
pdf.body('LLM 추론이 수 초 걸리는 반면 디스패치 결정은 밀리초 단위로 발생하므로, 주파수 분리 설계를 채택했다: LLM은 에피소드당 한 번(저빈도) 작동하고, 룰은 매 스텝(고빈도) L1에 의해 집행된다.')

pdf.subsection('하드 제약 및 생산 하한선')
pdf.body('L1 내의 하드 제약은 생산 달성률이 85% 하한선 아래로 떨어질 때마다 모든 활성 룰을 지워, 거버넌스가 긴급 로트 성능 추구 과정에서 주요 생산 목표를 희생하지 않도록 보장한다.')

pdf.subsection('데이터 흐름 및 학습 역학')
pdf.figure_placeholder('fig3_sequence_improved', '하나의 거버넌스 사이클에 걸친 계층 간 데이터 흐름. 룰은 학습 중에만 갱신되고 평가 중에는 동결된다.')
pdf.body('평가 중 룰이 동결되어 실증적으로 거의 발동하지 않는다. 즉, LLM은 런타임 필터라기보다 DRL 에이전트가 내재화하는 것을 형성하는 교사처럼 작동한다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(4, '실험 환경 및 설정')

pdf.subsection('디지털 트윈 및 로컬 LLM')
pdf.body('6개의 순차적 공정(A–F)과 임계 단계에서의 이종 장비를 갖춘 반도체 라인의 Siemens Tecnomatix Plant Simulation 모델을 사용한다. 본 모델은 실제 반도체 공장 데이터(실측 로트 도착 분포, 장비 처리 시간, 라우팅 제약)를 기반으로 구축되었으며, 소스 공장의 실제 처리량 및 사이클 타임 통계와 비교 검증되었다. Siemens Tecnomatix Plant Simulation은 반도체 및 전자 제조 분야에서 널리 채택된 ISO 9001 인증 상용 이산 이벤트 시뮬레이션 플랫폼으로, 라이브 공장 직접 실험의 안전·접근 위험 없이 고충실도 프록시를 제공한다.')
pdf.body('연구 환경 보안 정책으로 인해 외부 서비스로 생산 데이터를 전송하는 상용 LLM API를 사용할 수 없다; 이 제약은 실제 반도체 팹의 일반적인 데이터 거버넌스 요건도 반영한다. 따라서 Ollama 런타임을 통해 Mistral-7B를 L2 거버너로 로컬 배포한다. Mistral-7B는 (i) 구조화된 JSON 룰 생성에 충분한 지시 추종 능력, (ii) 에피소드당 추론 지연 5초 이내, (iii) 자유 재배포 가능성의 이유로 선택되었다.')

pdf.subsection('긴급 로트 시나리오')
pdf.body('소수의 긴급 로트("U"로 시작하는 식별자)가 라인 입구에 주입되어 가능한 한 빠르게 완료에 도달해야 한다. 원천 공장의 운영 매개변수에 기반하여 긴급 로트 비율을 전체 로트의 약 7.7%(258개 중 20개)로 설정한다. 이 밀도에서 긴급 로트는 병목 단계에서 빈번한 스케줄링 충돌을 유발하며 DRL 단독으로는 치명적 실패가 시작된다.')
pdf.body('시뮬레이터는 행동 9가 일반 대기열 순서보다 긴급 로트를 앞으로 이동시킬 수 있는 유일한 메커니즘이 되도록 구성된다. 이 설계 선택은 실험 유효성에 결정적이다.')

pdf.subsection('지표 및 실패 정의')
pdf.body('주요 지표는 긴급 로트 사이클 타임(CT): 긴급 로트의 평균 종단간 체류 시간(낮을수록 좋음). 치명적 실패 지표:')
pdf.equation('실패 지표 (식 1)', 'F_i = 1[CT_i > τ],  τ = 3,500s')
pdf.body('임계값 τ = 3,500s는 최종 공정 단계 로트의 최대 TAT(Turn-Around Time) 허용치로, 원천 공장의 운영 매개변수에서 파생된 사전적(a priori) 설정값이다. 경험적 CT 분포가 이 지점에서 자연적인 간격을 두고 두 개의 명확히 분리된 군집을 형성하여 사후적으로도 확인된다.')
pdf.equation('경험적 실패율', 'rho^ = (1/N) Σ F_i')

pdf.subsection('조건')
pdf.body('동일한 보상 함수를 공유하고 거버넌스에서만 다른 두 조건을 비교한다: (i) Pure DRL — L1/L2 계층 없는 Rainbow DQN; (ii) LLM-HRL (제안) — 전체 3계층 프레임워크.')

pdf.subsection('신뢰성 프로토콜')
pdf.body('각 조건을 5개의 독립 시드로 실행하고 실패율과 분산을 보고한다. 각 실행은 총 100 에피소드를 실행하며, 평가 지표는 마지막 30 에피소드(에피소드 71–100)에서 수집된다. 이 기간 동안 LLM이 생성한 룰은 동결되고 에이전트는 탐욕적으로 행동한다.')

pdf.subsection('에이전트 설정')
pdf.body('전술 에이전트는 숨겨진 레이어 [1024, 512, 256], 학습률 3×10-⁴, 할인 γ=0.98, 재플레이 용량 100k, 다단계 n=3, 300스텝마다 타겟 업데이트를 사용하는 Rainbow DQN이다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(5, '결과')

pdf.subsection('5.1 디지털 트윈 비결정성')
pdf.figure_placeholder('fig_nondeterminism', '동일한 설정과 시드를 가진 두 Pure-DRL 실행. 초기 학습 궤적이 거의 일치하다가 발산하여, 실패한(CT ~=3,706s)과 성공한(CT ~=2,274s) 평가로 끝난다.')
pdf.body('결과가 랜덤 시드에 의해 제어되지 않음을 확립한다. 동일하게 시드된 두 실행이 처음 몇 학습 에피소드 동안 거의 동일한 궤적을 공유하다가 발산한다. 동일한 명목 시드가 한 실행에서는 성공을, 다른 실행에서는 실패를 낳을 수 있다. 이는 결함이 제어 가능한 시드가 아닌 디지털 트윈에 내재적임을 의미한다.')

pdf.subsection('5.2 신뢰성: 실패율 및 분산')
pdf.table_2col(
    '주요 시나리오의 신뢰성 결과 (긴급 로트 20개, N=5 시드)',
    ['조건', '실패율', 'CT 평균', 'CT 표준편차', '최악'],
    [
        ['Pure DRL',       '60%',      '3,819s',      '914s',     '4,554s'],
        ['LLM-HRL (제안)', '0%',       '3,071s',      '248s',     '3,309s'],
    ]
)
pdf.body('p=0.010 이항검정;  ΔCT -19.6%;  분산 감소 73%')
pdf.figure_placeholder('fig_mechanism', '20개 긴급 로트에서의 학습 수렴(5 시드; 음영 띠 = ±1 SD). Pure DRL 평균은 높고 가변적인 반면, LLM-HRL은 안정적으로 수렴한다.')
pdf.body('Pure DRL은 실행의 60%에서 치명적으로 실패했고(rho^_DRL = 3/5), LLM-HRL은 어느 것도 실패하지 않아(rho^_LLM = 0/5), Δrho = 0.60을 달성했다. 귀무가설 H0: rho_LLM = 0.60 하에서 N=5번의 거버넌스 실행에서 k=0 실패를 관찰할 확률:')
pdf.equation('이항검정 (식 4)', 'p = Pr(X <= 0 | X ~ Bin(5, 0.60)) = (0.40)^5 ~= 0.010')
pdf.body('실패율 감소가 5% 수준에서 통계적으로 유의함을 확인한다. 평균 CT는 19.6% 개선되었고(3,819s → 3,071s), 사이클 타임 분산은 73% 감소했다(표준편차 914s → 248s). 생산 달성률은 조건 간 통계적으로 변화 없었다(p>0.05).')
pdf.figure_placeholder('fig_perseed', '주요 시나리오(긴급 로트 20개)의 시드별 평가 CT. 빨간 음영은 치명적 실패 영역(CT > 3,500s). Pure DRL은 세 시드(42, 2024, 7)에서 실패하는 반면 LLM-HRL은 모든 5개에서 임계값 아래 유지된다.')

pdf.subsection('5.3 거버넌스 하에서의 생산 달성률')
pdf.table_2col(
    '조건별 목표 달성률(%) — N=5 시드, 평균±SD. 통계적으로 유의한 차이 없음 (p>0.05)',
    ['긴급 로트', 'Pure DRL', 'LLM-HRL'],
    [
        ['12개', '88.2 ± 2.1', '88.0 ± 1.1'],
        ['20개', '87.2 ± 2.5', '86.1 ± 2.7'],
    ]
)
pdf.body('거버넌스는 처리량을 희생하지 않고 신뢰성을 향상시킨다.')

pdf.subsection('5.4 희소 거버넌스가 정책을 형성하는 방법')
pdf.body('LLM은 스텝의 단 몇 퍼센트에서만 개입하고, 평가 중 동결된 룰은 거의 발동하지 않는다. 그러나 거버넌스 에이전트는 여전히 낮고 안정적인 사이클 타임을 달성한다. 이는 거버넌스의 가치가 얼마나 자주가 아닌 언제 개입하느냐에 있음을 나타낸다. 긴급 로트가 존재하고 행동 9를 취해야 할 때의 소수의 시기적절한 마스킹이 에이전트에게 긴급 우선순위를 발동하는 행동을 가르치기에 충분하다.')

pdf.subsection('5.5 어블레이션: 적응형 LLM 거버넌스의 가치')
pdf.body('LLM의 적응형 룰 생성이 단순한 행동 제약의 존재와 구별되는 기여를 하는지 검증하기 위해 세 번째 조건을 도입한다: Rule-Based — LLM-HRL과 동일한 L1 집행 메커니즘을 사용하되, 에피소드 단위 LLM 수정 없이 고정 룰(urgent_lot_count > 2 → forbid {0-8})만 적용하는 거버넌스 에이전트이다.')
pdf.table_2col(
    '어블레이션 결과 (긴급 로트 20개, N=5 시드)',
    ['조건', '실패율', 'CT 평균', 'CT std', '최악'],
    [
        ['Pure DRL',       '60%',    '3,819s', '914s', '4,554s'],
        ['Rule-Based',     'TBD%',   'TBD',    'TBD',  'TBD'],
        ['LLM-HRL (제안)', '0%',     '3,071s', '248s', '3,309s'],
    ]
)
pdf.body('Rule-Based 조건이 LLM-HRL과 유사하게 낮은 실패율을 달성한다면, 어떤 형태의 행동 제약도 충분함을 시사한다. 반대로 Rule-Based가 높은 분산이나 실패를 보인다면, LLM의 적응적 KPI 기반 수정이 핵심 메커니즘임을 확인하는 것이다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(6, '배포 교훈: 로컬 LLM 룰 생성')
pdf.body('7B 로컬 모델을 제어 룰 생성기로 운영하는 것은 체계적이고 재현 가능한 오류 패턴을 드러냈다.')

pdf.subsection('룰 형식 및 대표 예시')
pdf.body('각 룰은 JSON 튜플 (metric, operator, threshold, forbidden_actions, rationale)이다. 전형적인 잘 형성된 룰은 urgent_lot_count > 2일 때 행동 9를 제외한 모든 행동을 금지하며, 공정 엔지니어를 위한 감사 추적 역할을 하는 인간이 읽을 수 있는 근거 문자열을 동반한다.')

pdf.subsection('오류 분류 및 완화')
pdf.body('608개의 룰 중 14.5%가 세 가지 유형의 결함을 포함했다.')
pdf.bullet('(1) 척도 혼동 (9.0%):', 'LLM이 에피소드 누적 KPI 값을 L1이 스텝별 스냅샷에 대해 평가하는 임계값에 복사하는 경우. 이러한 룰은 발동할 수 없어 거버넌스를 조용히 비활성화한다.')
pdf.bullet('(2) 역전 논리 (6.7%):', '모델이 높을수록 나쁜 지표에 <=/< 연산자를 사용하여 정상 조건에서 룰이 발동하는 경우. 행동이 거의 항상 마스킹된다.')
pdf.bullet('(3) 중요 행동 억제 (2.6%):', '모델이 긴급 우선순위 행동(행동 9) 자체를 금지하는 경우.')
pdf.body('두 단계로 완화했다. 프롬프트 재설계(명시적 스텝별 지표 범위, 올바른 예시, 긴급 행동 금지 금지 지시)가 오류율을 73.9%에서 12.0%로 줄였다. 이후 코드 수준 새니타이저가 잔여 오류를 결정론적으로 수정하여 구성에 의해 안전한 룰을 산출했다. 이 프롬프트+새니타이저 파이프라인을 신뢰할 수 있는 로컬 LLM 거버넌스를 위한 구체적이고 전달 가능한 "결함 제거" 레시피로 간주한다.')

pdf.subsection('대표 룰 패턴')
pdf.table_2col(
    '608개 생성 룰의 대표 패턴 (†집행 전 새니타이저가 탐지·수정. 발동률은 룰 동결 평가 기준)',
    ['패턴', '유형', '빈도', '발동률'],
    [
        ['urgent>2 → {0-8} 금지',          '정상',        '34%', '0.8%'],
        ['iplt_over>3 → {0-5} 금지',       '정상',        '23%', '1.1%'],
        ['urgent>0 → {0-8} 금지',          '정상(엄격)',   '15%', '4.3%'],
        ['shortage>450 → {6-9} 금지†',     '척도 오류',    '9%',  '—'],
        ['shortage<5 → {9} 금지†',         '역전 논리',    '7%',  '—'],
        ['urgent>1 → {9} 금지†',           '행동 억제',    '3%',  '—'],
        ['기타 / 복합',                     '—',            '9%',  '—'],
    ]
)

# ══════════════════════════════════════════════════════════════════════════
pdf.section(7, '논의')
pdf.body('결과는 LLM-over-DRL 거버넌스의 목적에 대한 재정의를 요구한다. 거버넌스 에이전트는 운이 좋은 Pure-DRL 실행보다 뛰어나지 않다; 둘 다 유사한 사이클 타임으로 수렴한다. 그 가치는 운이 나쁜 실행을 제거하는 데 있다. 제조 맥락에서 잘못 스케줄된 긴급 로트 하나가 품질 시간 위반과 수율 손실로 이어질 수 있는 경우, 실패 꼬리를 제거하는 것은 평균의 한계적 개선보다 가치 있는 경우가 많다.')
pdf.body('이 관점은 기여를 신뢰성 공학과 정렬시킨다. 거버넌스는 기본 비결정성을 제거하는 것이 아니라 그 해로운 결과를 마스킹하는 내결함성 메커니즘으로, 고전적 내결함성 시스템의 중복성이나 우아한 성능 저하와 유사하다.')
pdf.body('거버넌스 계층의 해석 가능성은 사회기술적 배포에서 중요하다. 각 룰이 명시적인 자연어 근거를 가지고 있어 공정 엔지니어가 시스템이 스케줄러를 제약한 이유를 감사할 수 있다. 거버넌스 오버헤드는 무시할 수 있는 수준이다: 에피소드당 LLM 호출 1번(약 2–5s), 일반적으로 2–4개의 룰, 마이크로초 단위의 L1 평가 비용.')

pdf.subsection('사회기술적 배포에 대한 시사점')
pdf.body('3계층 아키텍처는 자연스럽게 조직적 역할에 매핑된다: L0는 자동화된 디스패처, L1은 기계 속도로 절차적 룰을 집행하며, L2는 성능 데이터를 검토하고 에피소드당 한 번 지침을 수정하는 교대 감독자로 작동한다. 모든 거버넌스 결정은 에피소드 인덱스, 결정 시점의 KPI 스냅샷, 생성된 룰 집합, 그리고 LLM의 자연어 근거 문자열을 포함하는 구조화된 감사 기록으로 저장된다.')
pdf.body('보상 재가중(R_urgent 증가)의 자연스러운 대안은 두 가지 문제를 보였다: 더 높은 가중치가 TMR을 허용 가능한 하한선 아래로 밀어내고, 비결정성이 동일한 가중치를 일부 실행에서 도움이 되게 하고 다른 실행에서는 과도하게 만들어 분산을 증가시켰다. 보상 신호가 아닌 행동 공간에 작용하면 두 함정을 모두 피한다.')

# ══════════════════════════════════════════════════════════════════════════
pdf.section(8, '결론')
pdf.body("""
실제 반도체 공장 데이터 기반으로 구축된 고충실도 디지털 트윈 환경에서,
신뢰할 수 없는 DRL 스케줄러의 의존 가능성을 실질적으로 향상시키는
3계층 로컬 호스팅 LLM 거버넌스 HRL 프레임워크를 제시했다.
주요 시나리오(긴급 로트 20개, N=5 시드)에서 거버넌스는 치명적 실패율을
60%에서 0%로 감소시켰고(p=0.010, 단측 이항검정), 생산 달성률의 손실 없이
사이클 타임 분산을 73% 감소시켰다.
두 조건의 CT 분포는 선택한 실패 임계값 τ=3,500s에서 자연적인 간격으로
명확하게 분리된 군집을 형성하며, 이 결과가 임계값 선택의 인공물이 아님을 확인한다.
또한 거버넌스 효과가 런타임 제약이 아닌 DRL 학습 궤적 교정에서 비롯됨을 보였다.
7B 로컬 모델의 룰 생성 오류 패턴과 배포를 안전하게 만든 프롬프트-새니타이저
파이프라인을 추가로 보고했다.
LLM 거버넌스는 성능 최적화가 아닌 사회기술적 제조 시스템에서 학습 컴포넌트를
위한 내결함성 메커니즘으로 이해하는 것이 가장 적절하며, 고충실도 디지털 트윈과
에피소드 단위 LLM 감시의 결합은 실제 반도체 팹에서 신뢰 가능한 강화학습
배포로 나아가는 실용적인 경로를 제공한다.
""")

# ══════════════════════════════════════════════════════════════════════════
pdf.section(9, '한계 및 향후 연구')

pdf.subsection('통계적 검정력')
pdf.body('신뢰성 연구는 조건당 N=5 시드를 사용한다; 실패율 감소는 이항검정으로 통계적으로 유의하지만(p=0.010), 평균 CT 비교는 검정력이 부족하다(p=0.115, t-검정). N=10 시드로의 확장이 CT 평균에서 p<0.05를 달성할 것으로 기대되며, 이는 가장 시급한 후속 과제이다.')

pdf.subsection('디지털 트윈 충실도 대 실제 배포')
pdf.body('본 연구에서 사용된 디지털 트윈은 실제 반도체 공장 데이터를 기반으로 구축되고 실제 생산 통계와 검증되어 고충실도 실험 기반을 제공한다. 그러나 모든 실험은 시뮬레이션 환경에서 수행되었으며, 실제 팹으로의 배포를 위해서는 추가적인 통합 작업, 공장 장비와의 실시간 통신 인터페이스, 그리고 공식적인 안전 자격 인증(예: IEC 61508 SIL 평가)이 필요하다.')

pdf.subsection('단일 라인 범위')
pdf.body('모델은 6개의 순차적 공정 단계를 갖는 단일 라인을 나타낸다. 라인·제품·모델 패밀리 간 일반화는 평가되지 않았으며, 다른 팹이나 제품 조합에 방법론을 적용하려면 디지털 트윈 재보정이 필요하다.')

pdf.subsection('LLM 룰 품질')
pdf.body('로컬 7B 모델은 14.5%의 룰 오류율을 보였으며, 새니타이저 파이프라인이 런타임에서 잔여 오류를 0으로 감소시켰다. 더 큰 지시 모델로의 업그레이드 또는 도메인별 스케줄링 데이터에 대한 미세 조정이 원시 오류율 감소에 기여할 것으로 예상된다.')

pdf.subsection('보상 함수 동등성')
pdf.body('Pure DRL과 LLM 거버넌스 에이전트 모두 학습 및 평가 전반에 걸쳐 동일한 보상 함수를 사용한다. 따라서 관찰된 성능 차이는 보상 설계가 아닌 거버넌스 레이어에만 기인한다. 이 설계 선택은 의도적이며, 추가적인 보상 엔지니어링이 상호 보완적인 개선을 제공할 수 있음을 의미한다.')

# ══════════════════════════════════════════════════════════════════════════
# 참고문헌
pdf.section(0, '참고문헌')
refs = [
    'Moyne, J., & Iskandar, J. (2017). Big Data Analytics for Smart Manufacturing. Processes, 5(3), 39.',
    'Hu, Y., et al. (2019). Reinforcement Learning for Semiconductor Scheduling. ICCAD.',
    'Waschneck, B., et al. (2018). Deep Reinforcement Learning for Semiconductor Manufacturing. ETFA.',
    'Lee, D. C. (2025). A Digital Twin Framework for IPLT Optimization using RL and an LLM Agent. Master\'s thesis, Sungkyunkwan University.',
    'Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. AAAI.',
    'Pathak, D., et al. (2017). Curiosity-driven Exploration. ICML.',
    'Garcia, J., & Fernandez, F. (2015). A Comprehensive Survey on Safe Reinforcement Learning. JMLR, 16(1), 1437–1480.',
    'Du, Y., et al. (2023). Guiding Pretraining in RL with LLMs. ICML.',
    'Carta, T., et al. (2023). Grounding LLMs in Interactive Environments with Online RL. ICML.',
    'Liang, J., et al. (2023). Code as Policies. ICRA.',
    'Hasanbeig, M., et al. (2020). Cautious RL with Logical Constraints. AAMAS.',
    'Vezhnevets, A. S., et al. (2017). FeUdal Networks for Hierarchical RL. ICML.',
    'Avizienis, A., et al. (2004). Basic Concepts and Taxonomy of Dependable Systems. IEEE Transactions, 1(1), 11–33.',
    'Jiang, A. Q., et al. (2023). Mistral 7B. arXiv:2310.06825.',
    'Henderson, P., et al. (2018). Deep RL That Matters. AAAI.',
    'Agarwal, R., et al. (2021). Deep RL at the Edge of the Statistical Precipice. NeurIPS.',
    'Siemens Digital Industries Software (2023). Tecnomatix Plant Simulation (Version 2306).',
]
for i, r in enumerate(refs, 1):
    pdf.set_font('Malgun', '', 8.5)
    pdf.multi_cell(0, 5, f'[{i}] {r}')
    pdf.ln(0.5)

# ── 저장 ─────────────────────────────────────────────────────────────────
pdf.output(OUT_PATH)
print(f'저장 완료: {OUT_PATH}')
print(f'페이지 수: {pdf.page}')
