# 調査と境界判断

## Summary
旧BaseClient._register_trained_new_model (clients/base.py:286–324)の初期集計を数値部品として移植する。torch2.12.1+cpuの既存環境を継続し、新ライブラリ導入なし。

## Sources and Decisions
- 全体mean/varをtorchで計算しitem→Pythonfloat→M2倍算。逐次Welford/float64化は丸め順を変えるため採用しない。
- 全体n1M2.1、classn1M2zero、class昇順/欠落省略を維持。n1は旧未定義variance warningを生成せず同数値を返す。
- experiment.py:295–333の事前学習初期統計はWelford/到着順classで別契約。本関数をそこへ流用しない。
- sharedbackbone prepareは外部モデル側責務。prepare後のlossを入力にする。既存meanloss APIへ新per-sampleAPIを先取り追加しない。
- MNISTのNumPyラベルはint64だが、旧data/streams.py:28–43のgenerate_dataがtorch.FloatTensorへ変換する。今回の登録batchはCPUfloat32[N,1]であり、データ読込み直後のNumPydtypeと登録入力を区別する。
- 旧registerをSimpleNamespaceのmodel/clientと計算済みlossで直接呼ぶoracleを調査agentが実行。singletonとmulti-classの全体/class値を確認。
- 空batchがn0/meanNaN/M2.1を登録する事実をrootも再現。[LEGACY-007](../../../docs/research/implementation-findings/legacy-007-empty-batch-nan-initial-statistics.md)で追跡する。正常client/過去成果への影響は未確認。新入力契約は非空を要求する。

## Skills and Pre-write Gate
fable-method、kiro-spec-requirementsの事前ゲートを適用。9条件・EARS・入力異常・範囲と隣接契約を確認し、要件保存前にPASS。設計ではkiro-spec-designのlight discovery/generalization/adopt/simplificationを適用予定。

設計保存前に9条件の対応、具体ファイル・publicAPI・エラー・独立性・境界/依存・再検証契機と実行順を確認し、design-review-gateをPASS。既存型を返す一関数へ簡約し、torchの既存batch reductionを採用する。
