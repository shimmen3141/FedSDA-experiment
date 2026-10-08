# 候補検証sessionの保持と進行

| 文書 | 役割 |
| --- | --- |
| [brief.md](brief.md) | 境界の分割 |
| [spec.json](spec.json) | 承認revision/hashと進捗 |
| [requirements.md](requirements.md) | 要求 |
| [design.md](design.md) | 設計、旧処理との対応、検査の一覧 |
| [naming.md](naming.md) | source/testの名前 |
| [tasks.md](tasks.md) | 逐次3task |
| [research.md](research.md) | 旧処理の事実と判断 |
| [review.md](review.md) | 独立レビューと採否 |
| [mutation-and-cpu-evidence.md](mutation-and-cpu-evidence.md) | 変異とfresh CPUの証拠 |
| [integration-validation.md](integration-validation.md) | 全回帰・要求対応・未検証事項 |

状態: 要求r1・設計r3・命名r2・tasks r2は独立Luna承認済み。Task 1はHaiku 5.5、Task 3はLunaが承認。Task 2は指摘（変異1種の追加）を反映して再レビュー待ち。feature最終レビューは未実施。検証はWindowsの基準環境の結果で判定し、途中のWSLでの結果は参考として区別して記録している。各状態の正本はspec.jsonとtasks.md。
