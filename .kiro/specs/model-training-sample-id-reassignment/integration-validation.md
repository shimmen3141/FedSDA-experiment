# 実測: モデル学習標本のID付替え

## 範囲・環境

学習標本storeの単一ID付替えのみ。評価用stored_data/容量、counter、正式登録全体・client/通信/新全体runは後続。
2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。

## Task1

- RED: 新API未実装のAttributeError、-xで1 failed/2.51秒/exit1。
- GREEN: 新92件＋既存storage/sampler=242 passed/4.76秒/exit0。対象Ruff/format成功。
- 旧confirm36条件（空/非空元、先既存/新先/同ID、元欠落/各ID順）、signed4、両引数不正6型×元先有無48、過去snapshot/後続追加1、opaque/float64/meta payload3。
- 実旧train_data_storeのTensor参照/標本順/モデル順一致、空列先上書き・重複保持、列連結なし、過去snapshot構造保持と借用payloadの変更共有を確認。
- Pyright0 errors/0 warnings。実Luna独立242passed/3.72秒、静的検査/scan成功、Task1 APPROVED。主担当も実測/型/実diffを照合し完了。

## Task2

- test-only接続。production変更なし、RED N/A。対象104 passed/3.06秒、Ruff/format成功。
- class2/4×Adam標準/AMSGrad/SGD×共有更新有無の12条件各3回。初回更新後に実旧confirmと新標本storeのID付替えを実行し、上位でbinding IDを対応する。
- 毎回の抽出Tensor、loss、全parameter/grad、個別・共有optimizer state、明示Random終端をexact照合。previewはRandomのdeepcopyを使い、実更新の乱数を追加消費しない。3共有RNGとtorch defaultsの不変、過去snapshotの旧IDも確認。
- 実Luna独立348passed/6.61秒、静的検査/scan成功、Task2 APPROVED。指摘なし。

## Task3

- 対象＋既存AST: 768 passed/4.33秒。既存ownerの依存guardが適用され、guard追加・新REDはN/A。
- fresh新CPU: `../../venv/refactoring-tests/model_training_sample_id_smoke.py`成功。class2/4で標本store→ID付替え→sampler→3共同更新、旧snapshotとpayload参照保持、明示Random消費と両optimizer state、旧パッケージ非importを確認。
- 全pytest: 4746 passed/3 skipped/1既存warning、128.07秒、exit0。3 skipは既存Windows wrapper条件、warningは既存qint8 fixtureのTypedStorage。旧11条件・最終Residual Adapter＋Switching3条件の固定goldenを含む。
- JUnit: `../../venv/refactoring-tests/training-sample-id-full.xml`、4749 tests/0 failures/0 errors/3 skipped。
- Ruff全体成功、format117 files整形済み、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- `git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.py`は空。旧実装・goldenを変更していない。
- 要求/設計/命名のLF正規化hashは承認値と一致。検証済み実装commit `7886be9`。tracked Python＋2goldenの217パスをパス順、パスUTF8＋NUL＋CRLFをLFへ正規化した内容＋NULで連結したSHA256: `19594584f2ec90b73361f6a14e9b13f3e442201c8bea1b903039b13e62ef61cb`。

新client・新全体runのgolden一致を示す検証ではない。評価用標本store/容量・counter・正式登録全体の接続は後続。今回新たな旧正常経路の不具合は観測していない。

実Luna Task3 APPROVED。独立対象＋AST768passed/4.53秒と証拠一致、指摘なし。主担当完了gateは対象＋AST768passed/4.56秒、fresh新CPU/diff成功。

## feature最終判定

全task完了後の別実Lunaレビュー: GO。全8条件、契約/所有/上位接続/依存方向/設計/ファイル計画を確認し指摘なし。主担当判定VERIFIED。旧実装・golden・許容差を更新していない。
