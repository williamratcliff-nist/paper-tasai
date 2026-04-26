#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="pandoc-paper:local"
MAIN_MD="$ROOT_DIR/digital_discovery_paper.md"
SI_MD="$ROOT_DIR/TAS-AI_Digital_Discovery_SI.md"
COMBINED_MD="$ROOT_DIR/TAS-AI_arxiv_combined.md"
COMBINED_TEX="$ROOT_DIR/TAS-AI_arxiv_combined.tex"
COMBINED_PDF="$ROOT_DIR/TAS-AI_arxiv_combined.pdf"

strip_yaml() {
  awk '
    NR == 1 && $0 == "---" { in_yaml = 1; next }
    in_yaml && $0 == "---" { in_yaml = 0; next }
    !in_yaml { print }
  ' "$1"
}

cd "$ROOT_DIR"

echo "[build] Building container image: $IMAGE_NAME"
docker build -f Dockerfile.pandoc -t "$IMAGE_NAME" .

echo "[build] Writing combined arXiv markdown: $(basename "$COMBINED_MD")"
cat "$MAIN_MD" > "$COMBINED_MD"
cat >> "$COMBINED_MD" <<'EOF'

```{=latex}
\clearpage
\appendix
\setcounter{secnumdepth}{3}
\renewcommand{\thesubsection}{S\arabic{subsection}}
\renewcommand{\thesubsubsection}{S\arabic{subsection}.\arabic{subsubsection}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theequation}{S\arabic{equation}}
\setcounter{figure}{0}
\setcounter{table}{0}
\setcounter{equation}{0}
\setcounter{subsection}{0}
\setcounter{subsubsection}{0}
\section*{Supplementary Information}
\addcontentsline{toc}{section}{Supplementary Information}
```

EOF

strip_yaml "$SI_MD" >> "$COMBINED_MD"

echo "[build] Rendering combined arXiv LaTeX: $(basename "$COMBINED_TEX")"
docker run --rm -v "$ROOT_DIR:/work" -w /work "$IMAGE_NAME" \
  "$(basename "$COMBINED_MD")" \
  --filter pandoc-citeproc \
  --bibliography references.bib \
  --csl royal-society-of-chemistry-with-titles.csl \
  -s \
  -o "$(basename "$COMBINED_TEX")"

echo "[build] Compiling combined arXiv PDF: $(basename "$COMBINED_PDF")"
docker run --rm -v "$ROOT_DIR:/work" -w /work "$IMAGE_NAME" \
  --pdf-engine=xelatex \
  -V mainfont="DejaVu Serif" \
  "$(basename "$COMBINED_MD")" \
  --filter pandoc-citeproc \
  --bibliography references.bib \
  --csl royal-society-of-chemistry-with-titles.csl \
  -s \
  -o "$(basename "$COMBINED_PDF")"

echo "[build] Done"
ls -lh "$COMBINED_MD" "$COMBINED_TEX" "$COMBINED_PDF"
