# FedDrift固定ベースライン

`build_feddrift.py`は、既存実験のCSV・NPZから比較用のFedDriftベースラインを
データセット別に構築します。入力は`feddrift_sources.json`で管理し、日付付きの
結果パスをPythonコードへ埋め込みません。

新規構築:

```bash
python -m tools.baselines.build_feddrift --output results/baselines/feddrift_new
```

追加シード・掃引・データセットを既存ベースラインへ統合する場合は、追加結果を記載した
別のJSONを渡して`--extend`を使います。

```bash
python -m tools.baselines.build_feddrift \
  --source-config tools/baselines/additional_sources.json \
  --output results/baselines/feddrift \
  --extend
```

`--extend`は直接追記しません。既存baselineを一時ディレクトリへコピーし、重複・件数を
検証してからディレクトリ単位で切り替え、旧baselineを日時付きバックアップとして残します。
同じデータセット・シード・掃引軸・掃引値が一致する結果は重複、内容が異なる結果は競合として
扱います。
