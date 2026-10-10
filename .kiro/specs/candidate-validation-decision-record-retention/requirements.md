# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 最終構成のFedSDAの全体runを、新実装だけで実行でき、最終状態が実旧の全体runと一致する。次は、全体runの結果から、指標と、標本・イベントごとの列を導出して、旧の結果・goldenと照合する。その調査（2026-10-10）で、次のことが分かった。

- goldenの条件（sine2、client 3、標本1500件、集約間隔50。tests/test_proposed_regression.py の `COMMON`・`CASES`・`ALGORITHM`）を新の設定で表すと、新の全体runは、実旧の全体run（旧の設定の文脈の中で、旧の部品を旧の順に呼んだもの）と、サーバ・全client・診断の記録・乱数の最終状態まで一致する（下書きのtest。commitしていない）。
- goldenの比較項目のうち、候補の判定の列（`provisional_client_ids`・`provisional_positions`・`provisional_accepted`・`provisional_reasons`・`provisional_resolution_positions`）と指標（`provisional_proposal_count`・`provisional_forward_count`）は、旧のclientの `provisional_model_decisions`（候補の判定の一覧）から作られる。新のclientは、候補検証が確定するたびに、判定記録（`PostAlarmCandidateValidationDecisionRecord`、終端の回収では`IncompletePostAlarmCandidateValidationDecisionRecord`）を作るが、保持していない。適応記録は、棄却の理由（前半・後半のどちらの区間で不合格か）を持たないので、適応記録からは復元できない。
- 旧の最終構成（`forward_persistent`）で、候補の判定が記録されるのは、候補検証の確定（federated_drift_experiment/clients/fedsda.py 270行）と、終端での未完了の候補検証の回収（367行）の2か所である。

変えたいこと: clientが、候補検証の判定記録を、起きた順に保持する。保持した一覧が、実旧の `provisional_model_decisions` と、全項目で一致することを確かめる。

あわせて記録すること（調査の結果。このspecの成果物ではない）:

- Linuxの再現性（2026-10-08・10-10のユーザー決定による確認）: 旧実装の最終構成の3ケース（tests/test_proposed_regression.py の `compute_all`）を、WSL Ubuntu（Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu、1 thread）で、別のprocessで2回実行した。2回の結果（33指標、31の離散列のhash、経路の件数）は、完全に一致した。Windows用のgoldenとは、sine2（指標10、離散列15）とsea2（指標1、離散列2）で違い、mnist2は一致した。決定のとおり、Linux用のgoldenを、別ファイル・別testとして作る（後のspec）。
- (4b)「指標の導出とgoldenとの照合」を、3つに分ける: (4b-1)候補検証の判定記録の保持（本spec）、(4b-2)全体runの結果（指標と離散列）の導出と、実旧・Windows用goldenとの照合、(4b-3)Linux用のgoldenと、Linuxでの照合。
- goldenの指標のうち、計算量（`compute_*`の7項目）は、旧のclientが処理の各所で数える計数（`compute_counters`）から作られる。新実装には、この計数がない。(4b-2)では照合の対象から外し、移植するかどうかは、ユーザーへ確認する。

## Introduction

研究者が、全体runの後で、候補検証の判定（提案位置、採否、理由、確定位置ほか）の一覧を、clientから読めるようにする。

## Boundary Context

- **In scope**: 判定記録を起きた順に保持するowner。clientの組立てが、このownerを作ること。clientが、候補検証の確定と、終端での回収で、判定記録をownerへ足すこと。保持した一覧と、実旧の候補の判定の一覧との照合。
- **Out of scope**: 判定記録の中身と、判定そのもの（移植済み。変えない）。指標と列の導出、保存（次のspec）。最終構成で通らない、旧の判定の記録（時系列holdoutの方式の「標本が足りない」ほか）。
- **Adjacent expectations**: 標本1件の処理、候補検証の進行、適応記録の追加は、引数も挙動も変えない。

## Requirements

### Requirement 1: 判定記録の保持

**Objective:** As a 研究者, I want clientが、候補検証の判定記録を、起きた順に保持してほしい, so that 全体runの後で、候補の判定の列と指標を導出できる

#### Acceptance Criteria

1. When 標本1件の処理で、保持中の候補検証が確定したとき, the FedSDA Run Client shall その判定記録を、保持の末尾へ1件足す。
2. When 終端で、未完了の候補検証を回収したとき, the FedSDA Run Client shall その判定記録（未完了用）を、保持の末尾へ1件足す。
3. While 候補検証が確定していない標本の処理, the FedSDA Run Client shall 保持を変えない。
4. The 判定記録のowner shall 保持した記録を、足した順の、後からの追加で変わらない一覧として読める。
5. If 2つの判定記録の型のどちらでもない値を足そうとしたとき, the 判定記録のowner shall 保持を変えずに拒否する。
6. The clientの組立て shall clientごとに、空の、別々のownerを作る。

### Requirement 2: 実旧との一致

**Objective:** As a 研究者, I want 保持した一覧が、旧の候補の判定の一覧と一致してほしい, so that 次のspecで、候補の判定の列と指標を、旧と照合できる

#### Acceptance Criteria

1. When 同じ条件で、新と実旧のclientを進めたとき, the 保持した一覧 shall 実旧の `provisional_model_decisions` と、件数・順・全項目（提案位置、検出器名、採否、理由、区間の件数、学習の件数、検証の件数、比較した参照モデル、候補と参照の平均損失、後半の平均損失、確定位置、検証の種別、参照モデルの履歴の平均損失）で一致する。
2. The 照合 shall clientの全状態の新旧照合（clientの軌跡の対照と、全体runの対照の全条件）の中で行う。
3. The 全体runの対照 shall 候補の採用、候補の棄却、保有モデルの再利用または現行の維持、終端での回収、のそれぞれの判定を、少なくとも1条件で含む。含まないものがあれば、その事実を記録する。

### Requirement 3: 既存の保証を保つ

**Objective:** As a 研究者, I want 追加が、既存の挙動を変えないでほしい, so that これまでの対照が、そのまま成り立つ

#### Acceptance Criteria

1. The 追加 shall 判定記録の保持のほかに、clientの状態と乱数の消費を変えない（既存の対照が、そのまま通る）。
2. The 真の概念 shall 判定記録の保持に影響しない（真の概念を捨てる全体runと、本来の全体runで、保持した一覧が同じ）。
3. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、全体runで、判定記録が保持されることを確かめられる。
4. The 変更 shall 依存の許可の範囲に収める。
