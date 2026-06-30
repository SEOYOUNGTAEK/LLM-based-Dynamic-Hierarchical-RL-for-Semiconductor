# AAAI-27 논문 파일 동기화 규칙

## 파일
- `main.tex`      — 영문 제출본 (PDFLaTeX로 컴파일)
- `main_ko.tex`   — 한글 번역본, 내부 검토용 (**XeLaTeX 또는 LuaLaTeX**로 컴파일; `kotex` 사용)
- `references.bib`— 공용 참고문헌 (두 파일이 공유)
- `Figures/`      — 공용 피겨 (두 파일이 동일 파일 참조)

## 동기화 불변식 (반드시 유지)
두 파일은 **구조가 1:1로 동일**합니다. 한쪽을 수정하면 다른 쪽도 동일하게 갱신합니다.

| 항목 | 규칙 |
|---|---|
| `\section`, `\paragraph` | 개수·순서 동일 (제목만 번역) |
| `\label{...}` | **완전히 동일** (번역 금지) |
| `\cite/\citep/\citet{...}` | **완전히 동일** |
| `\includegraphics{Figures/...}` | **동일 파일·동일 너비** |
| 수식(`equation`), 수치, 통계값(p, CT, %) | **완전히 동일** |
| 산문(prose) | 한글본만 번역 |

## 수정 워크플로
1. `main.tex` 먼저 수정
2. 동일 위치를 `main_ko.tex`에 반영 (라벨/인용/수치는 복사, 산문만 번역)
3. 검증:
   ```
   diff <(grep -oE '\\label\{[^}]+\}' main.tex) <(grep -oE '\\label\{[^}]+\}' main_ko.tex)
   diff <(grep -oE '\\cite[pt]?\{[^}]+\}' main.tex|sort) <(grep -oE '\\cite[pt]?\{[^}]+\}' main_ko.tex|sort)
   ```
   둘 다 출력이 없어야 함 (동일).

## 피겨 재생성
`paper_aaai/generate_figures_aaai.py` 실행 후 `.pdf`를 `Figures/`로 복사.
데이터는 `paper_aaai/aaai_data.py`가 `output/`에서 라이브 로드 (20 seeds + ablations).

## 제출 트랙
AAAI-27 **Main Track** (기술적 신규성: ablation으로 거버넌스 메커니즘 분해).
