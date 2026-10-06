# 実測: モデル別学習・割当件数

## 環境・範囲

2026-10-07、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版/手順はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。
学習量・個別step・真concept診断件数の保持/加算移管/一回再編のみ。現在帰属ID・学習判断・全登録/新client/runは後続。

## Task1

- RED: 新module未実装のModuleNotFoundError、1 collection error/1.81秒、exit1。
- GREEN: 193 passed/1.84秒、exit0、Ruff/format成功。Pyright0 errors/0 warnings、exit0。
- 加算72条件（signed model/0・任意大int増分/None・signed concept/反復）、移管30条件（計数部分存在・先既存/新先・欠落元）、再編40条件（反復/連鎖/循環/多元先/0）、snapshot独立1、LEGACY011実再現/新拒否1、原子的入力拒否44、None/欠落時検査1、signed直接期待値4。
- 3辞書のkey順/概念内順と値を旧_attribute_model_training/_record/get_model_concept_counts/confirm/apply_server_mappingへ対照。旧同負ID通知の破損は実旧再現を残し、旧productionは変更しない。
- 初回Ruffでsnapshot変数をiterableの式とloopで兼用する警告を検出し、比較対象tupleを先に生成して修正。機能テストは初回189件から成功。
- 実Luna Task1 APPROVED、独立193passed/1.80秒、静的検査/scan成功、指摘なし。追加のTask2命名revision2もAPPROVED。

## Task2

- AST RED: 7 failed/719 passed/0.73秒、exit1。一般stdlib許可が通す依存を新exact guardで拒否する前の失敗。
- GREEN: 対象＋AST931 passed/4.42秒、exit0、Ruff/format/diff成功。18条件追加、dataclass/annotationsの公開symbolだけ許可し、module import・Counter/Random/数値/学習/他ownerを拒否。
- class2/4×Adam標準/AMSGrad/SGD×共有更新有無の12条件各3共同更新。初回後に実旧confirmと新加算移管、上位で標本/NN binding IDを対応。新samplerの独立Random preview順/件数を完了学習の計数へ渡す。
- 毎stepの全3計数/key順、抽出Tensor、loss、全parameter/grad/両optimizer、Random終端がexact一致。3共有RNG/torch defaults保持と過去snapshotも確認。
- `python -S ../../venv/refactoring-tests/model_training_and_assignment_counts_stdlib_smoke.py`成功。保持→加算移管→一回再編/copy分離、torch/numpy/旧非import。
- 実Luna Task2 APPROVED、独立対象＋AST931passed/8.43秒、静的検査/diff/scan成功、指摘なし。

## Task3

- fresh新CPU: `../../venv/refactoring-tests/model_counts_training_cpu_smoke.py`成功。class2/4で3回実学習→計数→初回後に加算移管→後続学習、両optimizerの蓄積stateと旧非importを確認。
- Ruff全体成功、format123 files整形済み、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- `git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.py`は空。旧実装・goldenを変更していない。
- 要求/設計/命名のLF正規化hashは承認値と一致。検証済み実装commit `d899fc5`。tracked Python＋2goldenの223パスをパス順、パスUTF8＋NUL＋CRLFをLFへ正規化した内容＋NULで連結したSHA256: `5a590a3629267daa24fb631c228f2ac4ad03dfa5eefaa4f197586894cd307833`。
- 全pytest: 5305 passed/3 skipped/1既存warning、177.27秒、exit0。3 skipは既存Windows wrapper条件、warningは既存qint8 fixtureのTypedStorage。旧11条件・最終Residual Adapter＋Switching3条件の固定goldenを含む。
- JUnit: `../../venv/refactoring-tests/model-counts-full.xml`、5308 tests/0 failures/0 errors/3 skipped。Lunaレビュー後にTask3をcheckする。

新client・新全体runのgolden一致を示す検証ではない。現在帰属ID・正式登録全体は後続。LEGACY011は不正同ID通知の再現のみで、正常経路・過去成果への影響は未確認。

実Luna Task3 APPROVED。独立対象＋AST再実行931passed/6.41秒、fresh/stdlib smoke成功、実測・hash・範囲を確認。主担当完了gateは931passed/5.67秒、両smoke/diff成功。指摘なし。

## feature最終判定

全task完了後の別実Lunaレビュー: GO。全9条件/契約/所有/上位接続/依存方向/設計/ファイル計画を確認。指摘なし、主担当判定VERIFIED。旧実装・golden・許容差は変更していない。LEGACY011は実旧再現と新拒否を追跡し、旧修正状態と通常経路への未確認事項を維持する。
