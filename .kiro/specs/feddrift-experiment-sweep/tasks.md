# Implementation Plan

- [x] 1. 既定完全条件からFedDriftを選択できるようにする
  - 旧の全6datasetの既定値と照合し、完全設定を作って既存構成入口へ接続する。
  - 条件の拒否とRNG不変、FedSDA構成の既存選択testが成功する。
  - _Boundary: 既定条件と構成選択_
  - _Requirements: 1.1, 1.2, 1.3, 4.3_

- [x] 2. 専用閾値の展開から実行・保存・再開まで接続する
  - 完全設定unionと専用軸、手法別run/保存/loader選択を既存掃引へ接続する。
  - mixed計画の順序とhash、不正軸、逐次/spawnの実結果と再開・不一致拒否が成功する。
  - _Boundary: 計画と実行・再開_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2_

- [x] 3. CLIと条件付き一覧を接続し新実装だけで確認する
  - 専用CLI引数、CSVのFedDrift条件、利用文書、freshと依存境界を接続する。
  - 実CLIで計画のみ/実行/再開/不正条件、一覧/記録が成功し、全回帰と品質検査を通す。
  - _Boundary: CLIと記録・新経路の統合_
  - _Requirements: 3.3, 4.1, 4.2, 4.3_

## Implementation Notes

- 要求gate: 数値ID/EARS、利用者の経路・不正条件・対象外を確認。設計gate: 全要求対応、既存ownerの再利用、依存方向とfile planを確認。task graphは1→2→3。ローカルreview off、実装前仕様群/全task後実装の独立CLIレビューを行う。

- 仕様レビューUTF-8再確認FAIL Major2件: blobsは既存新実装の対象外で6dataset維持という範囲を明記し、逐次/spawnの照合を完全設定・全指標・全表（環境/時間/保存先/実行状態を除外）へ明記。意味を追加せず既存境界を具体化。Minorのpending記録はレビュー中の正常状態として説明し、PASS後に更新する。最初の文字化けsessionは内容未確認のため承認に使わない。

- Task1: stub19 RED→対象と既存構成選択20 passed/2.34s。旧dataset既定のbootstrap6件と既存FedSDA hash sine2_interval3_seed0_98b98b1b6b40を確認。Task2: 軸未登録のTypeErrorでmixed test RED、対象38 passed/10.46s（逐次/spawn/全表比較/再開/不一致拒否）。初回tmp_pathのアクセス拒否は機能のREDに数えず、固定venv通常権限で確認。ローカルreview off、taskの実行証拠確認済み。

- Task3: 未登録CLI引数で1 RED→新FedDrift47件＋既存sweep51件の98 passed/66.98s。依存3091 passed/3.61s、fresh直接exit0/16flows、Ruff/format335、Pyright0、names6source未登録なし。依存登録scriptの正規表現で無条件returnが混入し827 failとなったが、実diffで原因を特定し対象登録だけ再生成して解消。既存sweep記録の期待値へ新軸Noneを追加（公開計画の既知の拡張）。型検査でhelperへの誤ったunion注釈とelseの絞り込みを修正。旧固定差分なし、変異未実施。

- 独立Haiku実装レビューPASS: session 72e91c8e-7ea4-4af6-ba3d-c22ca7b16d84、modelUsage claude-haiku-5-5/is_error=false、medium指定、process exit0。独立3189 passed/exit0、fresh直接exit0/16flows、Ruff/format335、Git状態不変。全pytest/Linux/Pyright/names/identity未再現。Minor1: sweep記録のplanにFedSDAでも専用軸nullが増えることは、承認済み新planのasdict記録として許容。run JSON/NPZ・FedSDAの値/hash/CSVは維持する。任意2: 構成名定数の導入は動作を変えないため不採用。確認3: 主担当は実装前（source基点021b0ef）に同条件のFedSDA hashを実行取得済み（task1のbootstrap）、独立担当は基点実行を未再現。再レビュー不要。

- レビュー後の最終品質: Ruff check/format335、Pyright0 errors/0 warnings/0 informations、pip check成功。WSL固定環境で対象47＋既存Linux golden4の51 passed/37.47s、exit0。旧golden更新なし。source/testを固定して全pytestを実行中。新規47件、前spec13607件から期待JUnit13654件。

- 最終全pytest: 13646 passed / 8 skipped / 9 warnings、724.80s、exit0。JUnit13654件（前spec13607＋新47）、failure/error各0。旧test_regressionとtest_proposed_regressionも成功。要求全4節/全3task、CLI/単一run/保存/再開/CSVの接続、設計境界と依存方向、blockedなしを確認。研究文書の旧dataset読込み名の誤記をget_dataset_specへ修正（契約変更なし）。

- 完了判定GO: 全要求/3task、独立仕様/実装レビュー、全pytestと新CLIのruntime liveness・統合・境界・blockedなしを確認。検証対象commit af8b6716053297534f5944b49b5b32ab35ba93bc、tracked Python＋2goldenの439パスのLF hash 06d691f387f1e2aed6b6c6b173964570c7be40961febcdb44335d7dd1b260db4。identityで旧固定差分なし・clean・JUnit成功を確認。完了記録は文書のみ。
