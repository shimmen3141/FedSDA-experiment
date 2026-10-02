# 設定基盤specの入口

## 正本と読む順序

1. `spec.json`: 現在の承認revision、次のtask、完了状態。再開前に必ず確認する。
2. `requirements.md`: 受け入れ条件の正本。初回だけでは全要求を完了しない。
3. `design.md`: 責務・依存方向・構築と検証の契約の正本。
4. `naming.md`: 現在の正式な名前・フィールド・値・配置の唯一の正本。revisionの承認を確認する。
5. `tasks.md`: 実行順・完了条件の正本。完了履歴を消さず、改名は追加taskで扱う。

上位方針は`../../../docs/research/refactoring-policy.md`と`../../steering/`に従う。
名前が他文書と食い違う場合は候補や履歴から選ばず、`naming.md`と`design.md`を照合して修正する。
承認不足のrevisionや未実装の後続計画を先取りしない。

## 補助資料（実装指示ではない）

| 文書 | 役割 |
|---|---|
| `review.md` | 時点ごとの実装・検証の証拠。過去の状態は現在の指示ではない |
| `research.md` | 設計調査の根拠 |
| `luna-review.md` | 過去の設計レビューと対応 |
| `luna-naming-review.md` | 命名レビューの履歴と採否 |
| `naming-reconsideration.md` | 前回案と比較・診断方式の候補表。正式採用はnaming.mdだけへ反映 |
| `history/` | 過去revisionの保存。旧名の実装・alias追加の根拠にしない |

初回の実行可能task終了後も後続計画4〜8は未承認・未完了として残る。
実験実行への接続は別specの契約を定義してから進める。
