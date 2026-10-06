# 実測: 損失統計のモデルID付替え

## 範囲・環境

既存統計storeの単一ID付替えだけ。モデル/標本/counterの付替え、現在帰属ID/保留解除を組み立てる正式登録全体、新client/全体runは後続。
2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。

## Task1

- RED: API未実装のAttributeErrorで63 failed/2.88秒/exit1。
- GREEN: 63 passed/1.94秒/exit0、対象Ruff/format成功。
- 実旧正式登録18ケース（負元/先既存・未登録/欠落/同ID/元0件）、汎用signedと同ID4ケース、両IDの不正型/元先有無40ケース、独立取得値と後続Welford更新1ケース。
- 内部ownerのみ更新、既存先位置維持/新先・同ID末尾、元欠落no-op、全クラス/全field、過去snapshotと返却値改変の非波及を確認。
- Pyright0 errors/0 warnings。独立Lunaも63 passed/2.11秒・Ruff/format/diff成功、Task1 APPROVED。指摘なし。主担当も実測GREENと型/境界を照合して完了判定。

## Task2

- test-only接続なのでproduction RED N/A。追加時に既存test末尾を新関数の後へ挿入した編集ミスでRuff未定義変数と4 failed/63 passedを観測。既存assertを元の関数へ戻して修正。production変更なし。
- 67 passed/2.63秒/exit0、対象Ruff/format成功。
- class2/4×delay1/2の4条件で実旧register→保留→生統計更新→ready→confirmと、新producer/loss/initializer/store/pending→現在統計取得→付替え→明示clear→変更先統計更新を照合。
- 全native parameter snapshot値を旧prefix明示対応で比較し、登録後モデル変更はsnapshotへ非波及。付替え後も保留record/snapshot/対応ID/ready状態を保持し、明示clearのみ解除。
- 統計全field/モデル順、parameter/grad参照と値、Python/NumPy/Torch乱数、既定dtype/device/gradmode保持を確認。
- 独立Lunaも67 passed/3.23秒・Ruff/format/diff成功、Task2 APPROVED。指摘なし、主担当も証拠と実diffを照合して完了判定。

## Task3

- 既存ASTと対象: 731 passed/4.80秒/exit0。新APIの依存追加なし、guard変更/新RED N/A。
- fresh CPU新processでclass2/4のproducer→loss→初期統計→store→pending/ready→付替え→明示clearを実行、旧package非import、exit0。
- Ruff成功/format115 files already formatted/pip check成功/diff-check成功、Pyright0 errors/0 warnings。固定748c3aaとの差分（旧production/両golden/旧回帰test）は空。
- source215 Python/両goldenのSHA256: `cfd392fe76a2a8958845adf556e4322d45f151e7360fedbdfcbff9ac574ce660`。Git追跡パスsort→UTF8相対path+NUL+LF正規化内容+NUL。要求/設計/命名の承認hash一致。
- 全pytest: 4559 passed/3 skipped/1既存warning/131.96秒/exit0。旧11条件・最終3条件のgolden回帰を含み、golden/許容差は変更していない。skipは既存Windows wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- JUnit loss-statistics-id-full.xml: 4562 tests/0 failures/0 errors/3 skipped。独立Luna対象＋AST731 passed/6.62秒、fresh新CPUと静的検査も成功。
- 新たな旧正常実装の不具合は今回の範囲では観測していない。
- 実Luna Task3 APPROVED。承認後の主担当gateも731 passed/3.30秒、fresh CPU/diff成功。全3task check後、feature最終レビューへ進める。

## 全8要件trace

|要件|証拠|
|---|---|
|1.1|実旧18ケース/クラス順と全field/非零sourceの更新接続|
|1.2|件数が大きい既存先へ0件元を上書き、先位置維持/全class置換|
|1.3|未登録先/同IDの末尾と他ID相対順、汎用signed4条件|
|1.4|空/元欠落×先有無で状態不変/空統計非生成|
|2.1|両ID×不正5型×元先有無40条件、snapshot全値/順序保持|
|2.2|過去取得と一覧保持/返却値改変非波及/変更先追加Welford|
|2.3|保留record/ID/snapshot/readiness保持、上位明示clearとの実旧4条件接続|
|2.4|3乱数/既定dtype・device・gradmode、fresh新CPU/既存AST/品質と固定golden gate|

```text
python -m pytest tests/refactoring/test_loss_statistics_model_id_reassignment.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/loss-statistics-id-full-20261006 --junitxml=../../venv/refactoring-tests/loss-statistics-id-full.xml
python ../../venv/refactoring-tests/loss_statistics_id_reassignment_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```
