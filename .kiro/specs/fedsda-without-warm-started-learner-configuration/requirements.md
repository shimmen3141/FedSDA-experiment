# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 基準版11（2026-10-22）から、警報の後の「新しいモデル」の仮説の符号長を、初期化し直した学習者とwarm startの学習者の予測的符号長の短い方＋ln 2で測る（`new_model_hypothesis_learners=reinitialized_and_warm_started_learners`。根拠は[NEW-006](../../../docs/research/implementation-findings/new-006-minimum-description-length-keeps-partially-changed-concept-in-current-model.md)）。基準版16の本規模の結果では、新しいモデルを選んだ警報の17.4%でwarm startの学習者が短く、MNIST・fmnistでは36〜62%（2026-10-29に主担当が集計）。ただし、版16でwarm startを外したときの精度は測っていない。
- 何を変えるか: 最終構成の既定から、warm startの学習者だけを外す（初期化し直した学習者だけで測る）構成を足し、構成要素の比較（`fedsda-component-ablation-v16`、`fedsda-component-ablation-new-datasets-v16`）へ追加で実行できるようにする。既定は変えない。
- 背景: 2026-10-29のユーザー判断（「specを追加し、①の実行が終わり次第その構成だけを追加で実行する。論文の補強としても有用」）。

## Boundary Context
- 対象: 新しい構成と、その構成を守るtest、文書、本規模の追加の実行のコマンド。
- 対象外: 既定の値（基準版16のまま。goldenの版を上げない）、学習者の計算、最小記述長の選択、ほかの構成、FedDrift・FedAvg、研究ツール。

## Requirements
### Requirement 1: warm startの学習者を外す構成
#### Acceptance Criteria
1. The FedSDA system shall 構成`fedsda-without-warm-started-learner`を持つ。組立ては、最終構成の既定から`training_data_assignment_settings.new_model_hypothesis_learners`だけを`reinitialized_learner`に替える（ほかの全部の設定は既定と同じ。`change_interval_search`は既定の新しい扱いのまま）。
2. When この構成で「新しいモデル」の仮説を測る, the FedSDA system shall 初期化し直した学習者だけの予測的符号長を使い、学習者の選択の符号長を足さない（学習者が1つのときの既存の扱い。ln 1 = 0）。
3. The FedSDA system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 新しい構成の設定が既定と`new_model_hypothesis_learners`の1か所だけ違うこと、構成の一覧に含まれ組立てが例外を出さないこと、goldenの事例にしない構成の一覧（`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 新しい構成の小さい全体run（条件は固定する: 基準版16のgoldenのケース`fedsda-sine2-adoption-without-validation`と同じデータ・seed・client数・標本数・間隔で、構成だけを新しい構成にする）で、新しいモデルが1回以上選ばれ、選択の記録の`new_model_learner_name`に`warm_started_learner`が現れないことを確かめる。学習者が1つのときにln 2を足さないことは、純粋な計算（`select_new_model_learner`）の部品のtestで確かめる（記録の表には前半の損失と切替の符号長の列が無く、全体runの記録から符号長を再計算できないため）。
3. The validation shall 基準版1〜16のgoldenの照合、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と追加の実行
#### Acceptance Criteria
1. The documentation shall 実験の入口（構成の一覧）、モデル再利用の文書の最小記述長の節、NEW-006に、新しい構成を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認（`--print-plan`）をして、構成要素の比較の2つのstudy（6データと新しい4データ、seed 0〜9、間隔4つ、10 client×5000標本）へ、この構成だけを追加で実行するコマンドをユーザーに渡す。結果は、同じstudyの`fedsda`（基準版16）とデータ・間隔・seedで対応させて集計し、NEW-006と実験の台帳に記録する。採否はユーザーが判断する。
