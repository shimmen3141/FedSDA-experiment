#!/usr/bin/env bash
# 研究用CLI専用。既存のPareto掃引wrapperとは引数と成果物の形式が異なる。
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
workers="${FDE_WORKERS:-64}"
out="${1:-results/fedsda-shadow-controls/$(date +%Y%m%d_%H%M%S)_uranium-s1-controls-seeds0-9-n5000}"
if [[ -e "$out" ]]; then
    echo "出力先が既に存在します: $out" >&2
    exit 1
fi
args=(--output "$out" --intervals 100 500 --seeds 0 1 2 3 4 5 6 7 8 9
      --sample-count 5000 --horizon-samples 240 --workers "$workers")
# 計画確認ではディレクトリを作らない。
python tools/research_shadow_learner_diagnostic/run_controls.py "${args[@]}" --print-plan >/dev/null
time_report="${out}.time.txt"
execution_log="${out}.execution.log"
mkdir -p "$(dirname "$time_report")" "$(dirname "$execution_log")"
echo "出力先: $out（120条件、既定との対照を含む240 run）"
/usr/bin/time -v -o "$time_report" \
    python -u tools/research_shadow_learner_diagnostic/run_controls.py "${args[@]}" \
    2>&1 | tee "$execution_log"
python tools/research_shadow_learner_diagnostic/analyze_controls.py "$out"
cat >"$out/README.md" <<'EOF'
# S1の無変化・オンライン継続対照

120条件（各条件の既定と診断で240 run）、5,000標本、seed0〜9、集約間隔100/500。
SEA4/SINE2の通常の変化と、SEA4の概念0/2・SINE2の概念0/1に固定した入力。
事前学習も固定概念と同じ。採用・製品の予測は変更しない。

分析はanalysis_controls.md、集計はsummary_controls.json、各条件の対照と診断はresults.json。
全標本の軌跡はJSONL、実行時の全ソースhashと条件はresearch-source.json。
ログとGNU timeはこのディレクトリ名に.execution.log/.time.txtを足したファイル。
条件数と入力源の照合・既定との一致・軌跡の再計算が成功してからこのREADMEを作成する。

研究記録: docs/research/fedsda-shadow-learner-s1-controls-20261024.md。
最終判断は、無変化時の交差と通常の変化時の交差、seedの偏り、オンライン継続との比較を確認して追記する。
EOF
echo "実験と照合が完了しました: $out/analysis_controls.md"
