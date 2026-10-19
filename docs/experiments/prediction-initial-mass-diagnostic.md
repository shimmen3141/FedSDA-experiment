# 新モデルの予測初期質量の診断

`--seeds 0-2`は`--seeds 0 1 2`と同じ。終端込みで、`0-2 5 8-9`の混在も可。各値は0以上2**32未満、重複・逆順は不可、展開後100000個以内。[seed指定の共通説明](experiment-sweep.md#seedの列挙と範囲)。

通常構成には追加しない研究用対照。[spec](../../.kiro/specs/prediction-initial-mass-diagnostic/design.md)。共通src・設定・pytest/CI/fresh・Ruff登録は変更しない。コードは`tools/research_prediction_initial_mass/`に閉じ、専用検証も明示実行だけ。既存DのCLIとは独立する。

|mode|意味|
|---|---|
|validation（V）|現行の候補検証あり|
|immediate（O）|現行の候補検証なし|
|ratio-preserved（R）|検証なし。新IDへ一様初期質量、既存ID間の比率を保持|
|small-new-mass（P）|検証なし。新IDの初期質量をRの1/Tへ抑え、既存比率を保持|

Tは既存のFixed-Share時間尺度を使う。新しい設定値は増やさない。初回、集合不変、削除のみ、全置換、残存質量0は元の処理。集約後のFIFO再較正や通常の重み更新は変更しない。O→Rで比率保持、R→Pで新ID質量の違いを調べる。共有バックボーンの干渉はこの介入では防げない。

## サーバでの本規模対照

6条件×3seed×4方式=72run。全て10clients、5000/client、FIFO/Tは既定設定から保存する。Linuxのcheckoutで次をtmux内から実行する。同時実験があればworkers合計14以下にする（例: 各7）。

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/research_prediction_initial_mass/run.py \
  --cases mnist2:100 mnist2:200 sea4:50 mnist4:500 circle2:200 sine2:500 \
  --seeds 0 1 2 --sample-count 5000 --workers 14 \
  --purpose '候補検証なしの低下に新モデルの初期予測質量が寄与するか、既存比率保持Rと小質量PをV/Oに対照し、精度・通信・学習経路を切り分ける。採否は自動決定しない。' \
  --output-directory results/fedsda-candidate-validation/prediction-initial-mass-n5000-seeds0-2
```

専用の保存処理であり標準sweep wrapperは使わない。既存ディレクトリ内の同名runは空でも拒否する。自動上書き・再開はない。結果は独自format `prediction-initial-mass-diagnostic-v1`のJSONと既存全記録NPZ。既存標準の結果形式を変更せず、既存解析CLIへ通常結果として渡さない。source/script hash、commit/dirty、RNG、介入前後重み、最終グローバルモデルの各parameterのdigestも保存する。

## 専用検証

```bash
python -m pytest tools/research_prediction_initial_mass/checks.py -q -p no:cacheprovider
```

小規模実験は1000/client・seed0の24runであり、本規模の採否を決める証拠ではない。72runは手元で自動起動しない。

手元24runは完了し、[結果と解釈](../research/fedsda-prediction-initial-mass-20261018.md)を記録した。Pの一律採用を支持しない結果であり、本規模のコマンドは確認用として扱う。

2026-10-19: 本規模72runも受領・解析した。[本規模の結果と判断](../research/fedsda-prediction-initial-mass-n5000-20261019.md)を参照。今回のPは候補検証の代替として採用を推奨しない。保存済みの同名出力先を使って再実行しない。
