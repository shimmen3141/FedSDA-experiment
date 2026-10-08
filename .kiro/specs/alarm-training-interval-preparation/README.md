# 警報時の学習区間の準備

前区間と変化区間への分割、前区間の評価標本保存→吸収を接続するspec。

## 正本と現在地

| 文書 | 役割 |
| --- | --- |
| [brief.md](brief.md) | 次の警報処理を分割した根拠と隣接境界 |
| [spec.json](spec.json) | 承認revision/hash、実装可否、進捗 |
| [requirements.md](requirements.md) | 要求r2、独立Luna承認済 |
| [design.md](design.md) / [naming.md](naming.md) | 設計r1・命名r4、独立Luna承認済 |
| [tasks.md](tasks.md) | tasks r4、承認済。7taskの進捗正本 |
| [review.md](review.md) | 独立レビュー証拠と指摘の採否 |
| [integration-validation.md](integration-validation.md) | 対象commit/hash、task・全回帰・品質・最終判定の証拠 |
| [mutation-evidence.md](mutation-evidence.md) | 6つの実source変異の検出とbyte復元 |

要求→設計/命名→tasks→実装→別feature最終GOを行う。命名承認前のsrc/test追加は行わない。後続の警報制御は、準備完了後の最小件数判定・区間解決・FIFO消費・検出器reset・記録/通知を担当する。

## 次の作業

Task 1〜6は独立GPT-6 Lunaレビューで承認・完了。現在はTask 7の全回帰を実測中。その後、別fresh reviewerのfeature最終GOを行う。次specは警報制御の残りで、本specの区間準備と既存の区間解決を呼出側へ接続する。
