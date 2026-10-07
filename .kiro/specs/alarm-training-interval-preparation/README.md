# 警報時の学習区間の準備

前区間と変化区間への分割、前区間の評価標本保存→吸収を接続するspec。

## 正本と現在地

| 文書 | 役割 |
| --- | --- |
| [brief.md](brief.md) | 次の警報処理を分割した根拠と隣接境界 |
| [spec.json](spec.json) | 承認revision/hash、実装可否、進捗 |
| [requirements.md](requirements.md) | 要求r2、独立Luna承認済 |
| [design.md](design.md) / [naming.md](naming.md) | 設計r1・初期命名r1、独立Luna承認済 |
| tasks.md（未作成） | 実装taskの正本 |
| [review.md](review.md) | 独立レビュー証拠と指摘の採否 |

要求→設計/命名→tasks→実装→別feature最終GOを行う。命名承認前のsrc/test追加は行わない。後続の警報制御は、準備完了後の最小件数判定・区間解決・FIFO消費・検出器reset・記録/通知を担当する。

## 次の作業

外部test下書きとAST走査でhelper/引数/局所名を洗い出し、初期命名へ追加してr2を独立レビューする。次にtask graphとtasksを独立レビューしてから実装する。現時点ではtask/実装は未着手。事前評価と吸収が同一登録分類器を参照する確認もtest条件に含める。
