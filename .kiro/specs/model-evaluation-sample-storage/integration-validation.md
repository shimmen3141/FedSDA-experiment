# 実測: モデル別評価標本の保持

## 環境と範囲

2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と手順はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。
評価標本の保持/抽出/ID対応のみ。学習標本・評価fallback・全登録/client/runは後続。

## Task1

- RED: 新evaluation package未実装のModuleNotFoundError、1 collection error/2.73秒、exit1。
- 初回GREEN途中で旧oracleのcurrent ID=999と欠落ID対応表999→12が衝突し、評価owner以外のmappingイベントに入ってfixture不足の9失敗となった。current IDを対応表外9999へ直し、Ruffのloop変数上書きを修正。production挙動の失敗ではない。
- GREEN: 290 passed/1.99秒、Ruff/format成功。144追加条件・18付替え条件・72再編条件、固定入力拒否13・操作拒否32・payload/snapshot3・empty/no-op Random検査8。
- 実旧処理を同じ開始Random stateで実行し、全ID順/標本順/Tensor identityと終端Random stateを照合。旧oracleは共有Python RNGをfinallyで復元する。
- Pyrightはsandbox実行で基準venvのnumpy/torchを解決できず失敗したため、従来の基準手順と同じ権限設定で再実行する。
- 通常権限での基準Pyright再実行は0 errors/0 warnings、exit0。
- 実Luna Task1 APPROVED。独立290passed・静的検査/scan成功、指摘なし。主担当も実diffと上記実測を確認。

## Task2

- AST RED: 16 failed/692 passed/0.69秒、exit1。新ownerのexact依存が未登録で実src走査と許可/禁止注入が失敗した。
- GREEN: 対象＋AST1000 passed/4.21秒、exit0、Ruff/format成功。ASTは44条件追加、2moduleへのmodule import・global random・private/child/再export・他ownerを拒否し、必要な公開symbolだけ許可する。evaluation packageは説明のみ。
- class2/4の実新旧NNに、追加→一回対応→単一付替え→2先への連結/両先超過再抽出→cat→損失評価を接続。全ID順/標本Tensor identity/Random終端/有界損失exact一致。過去snapshot、3共有RNG/torch defaults保持も確認。
- 旧oracleのmapperにはID変更イベント用fixtureを補い、該当する実旧経路を省略しない。新productionの他owner接続は追加していない。
- 実Luna Task2 APPROVED。独立対象＋AST1000passed/3.82秒、静的検査/diff/scan成功、指摘なし。
