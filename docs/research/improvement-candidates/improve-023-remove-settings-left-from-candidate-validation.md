# IMPROVE-023: 候補検証を既定から外した後に残った設定を整理する

- 日付: 2026-10-20
- 対象: `CandidateModelTrainingAndAcceptanceSettings.candidate_post_alarm_validation_sample_count`、`FedsdaRunClientScalarSettings.minimum_candidate_mean_loss_improvement`、`FedsdaRunClientSettings`の制約（候補の改善の最小幅＝早期終了の最小幅）、候補検証の処理（約1,900行、12モジュール）
- 基準commit: `053fefc`（基準版7）
- 種別: 簡略化（既定の動作を変えない整理。候補検証の処理を消す場合は、構成`fedsda-with-candidate-validation`と基準版1〜6の扱いが変わる）
- 状態: 未検証・未採用（候補検証の処理を消すかは、ユーザーの判断待ち）
- 優先: 4案の4番目（精度に影響しない。候補検証の処理を消すかの判断と一緒に行う）

## 前提（なぜ取り組むか）

2026-10-20に、既定を候補検証なしへ替えた。候補検証の処理は、主担当の判断で、構成`fedsda-with-candidate-validation`として残した（論文のablation、実旧との照合、基準版1〜6が候補検証の経路を通るため）。そのため、既定では使わない設定と、候補検証の名残の制約が残っている。

## コードで確認した事実

- `candidate_post_alarm_validation_sample_count`（既定10）: 検証なしの方針では使わない（設定の説明にも書いてある）が、必須の項目として既定の設定に残り、保存される。
- `minimum_candidate_mean_loss_improvement`（既定0.0001）: 候補検証の2区間の判定でだけ使う。既定では使わない。
- `FedsdaRunClientSettings`の制約: 「`minimum_candidate_mean_loss_improvement`を、早期終了の`minimum_validation_loss_decrease`と同じ値にする」。旧実装では1つの値（`NEW_MODEL_EARLY_STOPPING_MIN_DELTA`）を両方に使っていた名残。既定で候補検証を使わないなら、この制約の意味は無い。
- `maximum_tolerated_mean_loss_increase`（0.1）: 候補検証の参照の再適合の判定に使う。既定ではクラスタリング（`ModelClusteringCriteria`）とclientの再利用の判定（IMPROVE-021）の値として残る。
- 候補検証の処理: 検証の期間の状態、警報の先送り、4通りの判定、未完了の検証の締め、記録と指標。検証専用のモジュールは12個・約1,900行（入力の検査を多く含む）。ほかに26か所の分岐。

## 方針

案は2段に分かれる。

1. **候補検証の処理を残す場合（今の扱い）**: 名残の制約だけを外し、候補検証の設定を「候補検証ありの方針のときだけ意味を持つ」ことが分かる形にする（例: 方針が検証なしのとき、検証の標本数を記録の上で使わないと明記する）。既定の動作は変えない。
2. **候補検証の処理を消す場合（ユーザーが決める）**: 候補検証の処理・設定・記録・指標を消し、構成`fedsda-with-candidate-validation`を消す。基準版1〜6と、実旧の提案構成との照合は、候補検証の経路を通るので、その照合を残すか（旧実装を参照として残す限り要る）を一緒に決める。論文のablationの「候補検証あり」は、消す前の結果を使う。

## 懸念

- 設定の項目を消すと、保存した設定の形が変わり、runの識別子と、基準版の比べ方（後に足した・変わった項目の表）に波及する。
- 処理を消すと、候補検証ありの結果を新しいコードで再現できなくなる。

## 必要な比較と採否の基準

- 1は動作を変えない整理なので、全pytestと基準版の照合が値を変えずに通ることを確かめる。
- 2はユーザーの判断。消す前に、論文に要る候補検証ありの結果（本規模、同じマシン）がそろっていることを確かめる。

## 検証結果

（未記入）
