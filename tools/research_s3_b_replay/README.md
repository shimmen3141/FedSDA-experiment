# Bだけの計測付き再現（8run）

対象はMNIST4、間隔100/500、seed0/3/7/8、各client5000標本・10clients。
原Bの秒以外の全指標・イベント・入力署名を一致させた上で、保存済みWの全予測系列と比較する。
製品・元の研究実行コード・Wのproducerは変更しない。予測の戻り値を読む既存のW用trackerを再利用し、追加NN推論は0。

元B/WはLinux x86_64、Python3.10.16・NumPy2.2.6・Torch2.12.1+cpu。
手元WSLはPython3.14.4・NumPy2.4.6のため今回の再現には使わない。
環境が異なる場合は実験前に拒否する。サーバ実行・結果転送は従来どおりユーザー担当。

## uraniumでの実行

元のB/Wと同じvenvを使う。tmux内、refactor/architectureブランチで以下を実行する。
参照結果一式（baseline/runs/metadata等）と、W結果一式（predictionsを含む）が必要。
fullscale直下に複数の完了済み日時ディレクトリがある場合は--referenceを具体的な1件へ変更する。

```bash
git pull --ff-only origin refactor/architecture
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m tools.research_s3_b_replay.run \
  --reference results/fedsda-s3-fullscale \
  --warm-output results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000 \
  --output results/fedsda-s3-b-replay/20261025_uranium-mnist4-i100-500-seeds0-3-7-8-n5000 \
  --seeds 0 3 7 8 --intervals 100 500 --workers 64

python -m tools.research_s3_b_replay.analyze \
  results/fedsda-s3-b-replay/20261025_uranium-mnist4-i100-500-seeds0-3-7-8-n5000 \
  --reference results/fedsda-s3-fullscale \
  --warm-output results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000
```

`START B=0/8 reused_W=8`、完了ごとに`B 3/8 ... original_match=PASS`を表示。
64workers指定でも同時processは最大8。中断時は同じrunコマンドへ`--resume`を追加する。
完了済みcheckpointも毎回原B・系列・hashと再照合する。入力・ソース・環境・参照の不一致は拒否し、古い結果を上書きしない。
実行中は数値ソースを変更せず、同じ出力先へ同時実行しない。
内部確認用--print-planは読み取り専用。出力先を作らず、実行時の環境検査のみ延期する。

## 出力と解釈

- metadata.json: ソースhash、全条件、環境、原D/Bと保存済みWのresults/metadata hash。
- runs/、predictions/: 検証済みBと全予測。failures.json、attempts.jsonl: 失敗・再開の記録。
- analysis_comparison.json: 全seedの標準指標、W/Bの累積差・区間差・概念別差、元系列hash。
- analysis_comparison.md: D/W/Bの精度・既定差pp・W差pp・上り/下りMB・GMAC・秒、平均累積差、概念別救済/悪化数。

区間は既定で各client100標本。--window-sizeで変更できる。
全予測のclient/sample key、画像hash、観測label/concept、正誤を照合する。
累積差の末尾が保存済み全体精度差と一致することも確認する。
概念別精度差は各seed等重み、救済/悪化数は全seedの標本数合計。
標準Dの秒数はN/A。Wと新Bで負荷や計測処理が異なるため、時間差を方式固有の速度差とは呼ばない。

seedは結果を見て選んだ診断ケースであり、新規の性能検証ではない。
この計測で「差がいつ、どの概念の標本で生じたか」は分かるが、個々の候補や同期処理への因果帰属までは行わない。
真の概念を採用判定へ使わず、製品への採用・既定化も行わない。

## 手元での検証

```powershell
../../venv/Scripts/python.exe -m tools.research_s3_b_replay.checks
../../venv/Scripts/python.exe -m ruff check --isolated --no-cache --select E4,E7,E9,F,I tools/research_s3_b_replay
../../venv/Scripts/python.exe -m ruff format --isolated --no-cache --check tools/research_s3_b_replay
```

検証記録は[研究記録](../../docs/research/fedsda-s3-b-replay-20261025.md)。

## 到着結果

2026-10-26、B8run完了・失敗0、原B全非秒指標一致。[結果・判断](../../docs/research/fedsda-s3-b-replay-results-20261026.md)。ユーザー配置先はresults/fedsda-s3-b-replay直下。解析時はそのパスを指定する。4seedの原因診断であり、10seedの新しい性能検証ではない。
