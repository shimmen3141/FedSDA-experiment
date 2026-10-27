# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAを開発し、コードを公開する研究者（ユーザー）。
- 今の状況: 2026-10-20から既定は候補検証なし（`current_model_first_reuse_then_candidate_adoption_without_validation`）だが、警報の後の候補検証の処理（検証の開始・標本の観測・2区間の判定・警報の保留・
  未完了の検証の締め・記録・指標、中心の14モジュールで約2,000行）と、構成`fedsda-with-candidate-validation`が残る。論文では候補検証に触れない（2026-10-28のユーザー判断）。
- 何を変えるか: [IMPROVE-023](../../../docs/research/improvement-candidates/improve-023-remove-settings-left-from-candidate-validation.md)の案2。候補検証の処理・設定・記録・指標・構成を消す。
  候補検証を使わない計算（予測・学習・通信・計算量・乱数）は変えない。保存の形とFedSDAのrunの識別子の変更は受け入れる。基準版15を足す。

## Introduction
影響範囲の調査（gpt-6.1-sol、2026-10-28、読取りだけ、HEAD `0d6c586d`）の要点。根拠の行は調査の報告（[scoping-report.md](scoping-report.md)）と設計書。
- 検証ありの方針の値は`current_model_first_reuse_then_two_segment_candidate_validation`の1つ。選ぶ入口は構成`fedsda-with-candidate-validation`、設定の直接の組立て、golden helper、旧実装との照合のfixture、研究用の道具。
- 候補検証の処理は、検証が無いとき（sessionが`None`、holderが空）にすぐ戻る。検証の開始の側（参照モデルの固定）だけがtorchの乱数を使う。検証なしの経路で、その外の乱数の消費は見つかっていない（静的な追跡。実測は要求5）。
- `runtime/post_alarm_candidate_validation_session_start.py`は、検証なしでも使う候補の学習（`TrainedAlarmIntervalCandidate`、`train_candidate_on_alarm_change_interval`）を含む。丸ごとは消せない。
- golden: 版1〜6はFedSDAの12〜14 caseが検証ありの方針、版7〜14は4 case（`fedsda-sine2-events`、`fedsda-mnist2-decisions`、`fedsda-sine2-reference-reuse`、`fedsda-sine2-alternative-reference-reuse`）が検証ありの方針で、判定の記録を持つ。
- 保存の形: 検証なしでも空の記録の表（`candidate_validation_decision.*`の16列）と、0件・未定義の指標を保存する。設定の保存は全fieldを含み、runの識別子は構成名と保存した設定全体のhash。

ユーザーの決定（2026-10-28、主担当の推奨のとおり「その方針で進めてください」）:

| 判断 | 決定 |
|---|---|
| 過去のgolden | 保存済みのJSONは変えない。各版で候補検証ありの方針を使わないcaseだけを流し、消した項目は比べるときに名前を挙げて除く |
| 基準版 | 版15を足す |
| 旧実装との照合 | 候補検証ありの経路の照合だけをやめる。検証なしの変種と部品ごとの照合は残す |
| 研究用の道具 | 候補検証のモード・入口だけを外し、道具自体は残す |
| `fedsda-without-candidate-validation`の名前 | 変えない（候補の共有部の固定を外す構成として残す） |
| 旧実装（`federated_drift_experiment/`） | 変えない（`tests/test_proposed_regression.py`など旧実装自身のgoldenは残す） |

testは、重要なもの・後で効くもの・複雑なものに絞る（基準版12のspecと同じユーザー指示）。

## Requirements

### Requirement 1: 構成と設定の入口を閉じる
1.1 The configurations shall `fedsda-with-candidate-validation`を消す（定数・登録の一覧・組立ての分岐）。
1.2 The settings shall 検証ありの方針の値と、候補検証だけが使う設定（`candidate_post_alarm_validation_sample_count`、`minimum_candidate_mean_loss_improvement`、採否の方針の項目）を消す。
   採否の方針が1つだけになる型・項目は、型と項目ごと消す。
1.3 The settings shall 候補検証のための制約（MDLと検証ありの組合せの拒否、共有部の固定と検証ありの組合せの拒否、候補の改善の最小幅＝早期終了の最小幅）を消す。
1.4 The settings shall 候補検証と共有されている値（`maximum_tolerated_mean_loss_increase`、早期終了の`minimum_validation_loss_decrease`とその既定値0.0001、候補の共有部の更新の方針、警報の区間の選び方ほか）を残す。
   名前が候補検証を指す定数は、残る用途の名前に直してよい（値は変えない）。

### Requirement 2: 候補検証の実行と状態を消す
2.1 The runtime shall 候補検証の開始・標本の観測・判定・解決・警報の保留・未完了の検証の締め・参照モデルの固定と、clientの状態（session holder、判定の記録の保存、検証へ渡した標本の概念IDの保持）を消す。
2.2 The runtime shall 候補の学習（`TrainedAlarmIntervalCandidate`、`train_candidate_on_alarm_change_interval`ほか検証なしで使うもの）を、候補検証の名前のないモジュールへ移し、処理・乱数の消費の順を変えない。
2.3 The execution shall 実行の契約から、未完了の候補検証の締めの操作と段（`incomplete_candidate_validation_finalization`）を消す（FedAvgの何もしない実装を含む）。

### Requirement 3: 記録と指標を消す
3.1 The run records shall 記録の表`candidate_validation_decision.*`（16列）を消す。
3.2 The run metrics shall `candidate_validation_decision_count`、`candidate_validation_decision_metrics`（`CandidateValidationDecisionMetrics`）、適応の指標`start_candidate_validation_count`・`defer_during_candidate_validation_count`・`reject_candidate_model_count`を消す。
3.3 The run records shall 候補検証だけの適応の結果の値（`alarm_during_candidate_validation`、`alarm_interval_candidate_validation_started`、`post_alarm_validation_*`）と判定の理由の値を消す。採用・維持・再利用の値と指標は残す。

### Requirement 4: golden・照合・基準版
4.1 The goldens shall 版1〜14の保存済みのJSONを変えない。各版で、候補検証ありの方針を使うcaseを照合から外し、外したcaseの名前と理由をtestに明示する。
4.2 The goldens shall 残すcaseの照合で、消した記録の表・指標・設定の項目を、名前を挙げた一覧で除いてから比べる（一覧にない差は失敗にする）。
4.3 The run provenance shall `BEHAVIOR_BASELINE_VERSION`を15にする。版15のcaseは、版14のcaseから候補検証ありの4 caseを除いたもの。版15のgolden（Windows・Linux）は主担当が作る。
4.4 The tests shall 版15の各caseの、4.2で除いた後の値が、版14の同じcaseと一致することを確かめる。
4.5 The tests shall 候補検証ありの経路の、旧実装との照合（調査の報告の3の表の関数）を消す。検証なしの変種（`fedsda-without-candidate-validation`ほか）と部品ごとの照合は残し、検証ありのfixtureに依っていたものは、旧実装が同じ振る舞いになる検証なしの設定へ直す（直せないものは、理由とともに消す）。

### Requirement 5: 数値の不変
5.1 The change shall 候補検証を使わない全構成で、予測・学習したパラメータ・通信・計算量・runの指標（消したものを除く）・乱数の状態を変えない。4.1・4.2・4.4のgoldenの照合で確かめる。
5.2 The change shall FedDrift・FedAvgの保存の形とrunの識別子を変えない。

### Requirement 6: 研究用の道具と文書
6.1 The tools shall `run_candidate_validation_waiting_diagnostic.py`と`tools/research_candidate_validation_waiting/`を消す。`tools/research_prediction_initial_mass/`・`tools/research_provisional_candidate_expert/`は、候補検証のモード・変種だけを外し、
   残るモード・変種の専用の検証が通るようにする。移した候補の学習の関数をpatchする道具（`tools/research_s3_contribution/`、`tools/research_output_layer_learner/`ほか）を新しい場所へ追随させる。
6.2 The docs shall 今の状態の正本（`docs/overview/proposed-method.md`、`docs/experiments/experiment-sweep.md`、`docs/experiments/refactoring-baseline.md`の版の表、改善候補の一覧のIMPROVE-023の行）を直し、
   IMPROVE-023に日付つきの結果の節を足す。過去の記録・既存のspecは書き換えない。`docs/components/new-model-creation.md`・`model-reuse.md`は、今の実装の説明の部分だけを直す。
6.3 The docs shall 過去の結果（基準版14までの保存の形・識別子）を読む集計の道具が、今までどおり使えることを書く（runの対応はデータ・間隔・seedで取る）。
