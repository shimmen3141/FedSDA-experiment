# 警報応答の適応記録

| 正本 | 役割 |
| --- | --- |
| [brief.md](brief.md) | 境界の分割 |
| [requirements.md](requirements.md) | 観測可能な要求 |
| [design.md](design.md) | 記録ownerと完了情報の変換 |
| [naming.md](naming.md) | 実装・testの命名 |
| [tasks.md](tasks.md) | 逐次3task |
| [research.md](research.md) | 旧処理と通知先の調査 |
| [spec.json](spec.json) | 承認と進捗 |
| [review.md](review.md) | 独立レビューと採否 |
| [integration-validation.md](integration-validation.md) | 検証対象と証拠 |

Task1/2は独立承認済み。全回帰の証拠はintegration-validation.md、Task3レビューと別fresh feature最終GOは進行中。承認の正本はspec.json。

警報応答5結果の記録・切替位置・再利用件数を一つのownerへ保存する。候補検証の到達時/終端の記録、session保持、予測診断通知、新全体runの接続は後続specで扱う。
