# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 最終構成のFedSDAの全体runを、新実装だけで実行できる（fedsda-run-participant-preparation）。しかし、単一runの実行の枠は、clientへ真の概念を渡さないので、全体runでは、真の概念に依存する診断（真の概念別の診断証拠、モデルごとの割当概念の計数、標本ごとの記録の概念、クラスタリングの真の概念の一致）が行われない。旧は、これらを、runごとに計算して保存している。

変えたいこと（2026-10-10、ユーザー決定）: 実行の枠が、標本ごとの真の概念を、診断専用の引数として、clientへ渡す。真の概念が、判断（予測、検出、学習、モデルの選択）へ影響しないことを、testで保証する。

調査で確かめたこと:

- 実行の枠（`runtime/single_run_execution.py`の`execute_stream_protocol_run`）は、概念列（`ClientConceptTrace`。評価用の真値）を生成し、そこから観測列を作り、観測列だけを、区間の進行（`execution/stream_protocol_execution_loop.py`の`run_stream_protocol_intervals`）へ渡す。区間の進行は、clientの`process_observed_sample(observed_sample, sample_index)`を呼ぶ。
- 実行の枠の契約（`execution/run_participant_contracts.py`の`RunClientOperations`）と、そのtest（tests/refactoring/test_single_run_execution.py）は、「観測処理が、真の概念を受け取らない」形である。single-run-executionの要求1.4は、「真の概念・ドリフト位置を評価用の情報として区別し、通常の学習・予測に利用できる観測値と混同しない」。
- `FedsdaRunClient.process_observed_sample`は、任意の引数`evaluation_concept_id`を、すでに受け取れる。渡されたときだけ、真の概念に依存する診断を行う。渡したときに、診断まで実旧と一致することは、fedsda-run-participant-preparationの対照test（test専用の中継で渡す）で確かめてある。
- 旧のclientは、標本ごとに、観測値と一緒に、真の概念を受け取る（`process_one_step(x, y, concept_id)`）。

## Introduction

研究者が、全体runで、真の概念に依存する診断を、旧と同じように得られるようにする。真の概念は、観測値とは別の、診断専用の引数として渡し、判断へ影響しないことを保証する。

## Boundary Context

- **In scope**: 実行の枠の、clientの標本処理の契約（診断専用の真の概念の引数を足す）。区間の進行が、概念列から、標本位置の真の概念を引いて、clientへ渡すこと。全体runの実行が、概念列を、区間の進行へ渡すこと。既存のtestと共用scriptの、契約の変更への追随。真の概念が、診断以外の状態へ影響しないことのtest。
- **Out of scope**: 真の概念に依存する診断の中身（移植済み）。診断の保存と集計（次のspec）。真のドリフト位置を使う指標（次のspec）。サーバの操作の契約。
- **Adjacent expectations**: 概念列と観測列の生成、clientの標本処理、サーバの同期は、中の処理を変えない。観測標本（`ObservedSample`）は、真の概念を持たないままにする。

## Requirements

### Requirement 1: 真の概念の受渡し

**Objective:** As a 研究者, I want 実行の枠が、標本ごとの真の概念を、診断専用の引数としてclientへ渡してほしい, so that 全体runで、真の概念に依存する診断が得られる

#### Acceptance Criteria

1. When 区間の進行が、clientへ標本の処理を求めるとき, the Stream Protocol Execution shall 観測標本と標本位置に加えて、そのclientの概念列の、その標本位置の真の概念IDを、診断専用の引数として渡す。
2. The Stream Protocol Execution shall 真の概念を、観測標本の中へ入れない（観測標本は、特徴とラベルだけを持つ）。
3. When 全体runを実行するとき, the Single Run System shall 生成した概念列を、観測列と同じclientの順で、区間の進行へ渡す。
4. If 概念列が、決まった型でない、clientの数や順が観測列と合わない、または標本の数が観測列と違うとき, the Stream Protocol Execution shall どのclientの操作も呼ぶ前に拒否する。

### Requirement 2: 判断への不干渉

**Objective:** As a 研究者, I want 真の概念が、診断にだけ使われてほしい, so that 評価用の真値が、手法の判断へ漏れない

#### Acceptance Criteria

1. When 真の概念を渡した全体runと、渡さない全体run（clientが、真の概念を受け取らない）を、同じ条件で実行したとき, the 全体run shall 真の概念に依存する診断（真の概念別の診断証拠、割当概念の計数、標本ごとの記録の概念、保留中の標本と候補検証へ渡した標本に付く概念、クラスタリングの真の概念の一致）を除いて、同じ状態になる（予測、警報、適応、保有モデルとそのパラメータ、損失統計、学習データ、評価標本、通信量、クロス評価の記録、クラスタとID対応、乱数の最終状態）。
2. The 実行の枠の契約 shall 真の概念の引数が、診断専用であることを、名前と説明で示す。

### Requirement 3: 実旧との一致

**Objective:** As a 研究者, I want 全体runが、test専用の中継なしで、診断まで旧と一致してほしい, so that 次のspecで、診断を含む結果を、旧と照合できる

#### Acceptance Criteria

1. When 同じ固定条件と設定を与えたとき, the 全体run shall 実行の枠の本来の経路（test専用の中継を使わない）で、実旧の全体runと、各clientの全状態（真の概念別の診断証拠、割当概念の計数、標本ごとの記録を含む）、サーバの全状態、クラスタリングの診断の記録（真の概念の一致を含む）、乱数の最終状態が一致する。
2. The 対照 shall fedsda-run-participant-preparationの対照の全条件を、そのまま通す。

### Requirement 4: 既存の利用箇所と、新実装だけで動くこと

**Objective:** As a 研究者, I want 契約の変更が、既存の検証を弱めないでほしい, so that 実行の枠の、これまでの保証が残る

#### Acceptance Criteria

1. The 実行の枠のtest shall 区間の進行が、clientごと・標本位置ごとに、概念列の値を渡していること（clientと位置を取り違えないこと）を確かめる。
2. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、全体runで、標本ごとの記録の概念が、概念列と一致し、真の概念別の診断証拠が作られることを確かめられる。
3. The 変更 shall 依存の許可の範囲に収める。
