# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAを研究し、コードを公開する研究者（ユーザー）。
- 今の状況: 保有モデルの学習batchの抜き出し方（全履歴から一様＝既定、直近K件、指数減衰α）を、設定と掃引の軸で選べる（[held-model-training-sample-recency](../held-model-training-sample-recency/tasks.md)）。server本規模の比較で、直近を重く抜き出すと、局所の予測ではseaで上がるがmnistで下がり、同期の時点の予測と組み合わせるとseaの効果も消えた（研究バックログの7）。採る理由が無い。
- 何を変えるか: 2026-10-15、ユーザーの判断で、抜き出し方の選択肢（設定、抜き出しの処理の分岐、掃引の軸）を外し、コードを簡単に保つ。全履歴から一様に抜き出す動作と値は変えない。

## Introduction
外した実装は、gitの履歴（be9396b・7831e74）とspecに残る。再び試すときは、それを参照する。

## Boundary Context
- 対象: `LocalTrainingSettings`の3つの項目と検査、共有の設定の検査の`int | None`・`float | None`の扱い（この2項目のためだけに足した）、batchの抜き出しの処理の選び方の引数と直近K件・指数減衰の分岐、共同学習からの受渡し、clientの設定のKの検査、掃引の軸（計画の項目、文字列の解析と表示、コマンドの引数、一覧の表の列）、goldenのhelperの登録、test、文書。
- 対象外: 全履歴から一様に抜き出す動作、帰属の保留の容量の軸、ほかの構成。結果のディレクトリの分析script（gitの対象外。前の結果の表の列は、そのまま読める）。

## Requirements
### Requirement 1: 選択肢を外す
**Objective:** As a 研究者, I want 採らないことにした抜き出し方の選択肢がコードに残らないこと, so that コードを簡単に保てる。
#### Acceptance Criteria
1. The FedSDA system shall `LocalTrainingSettings`に、抜き出し方の3つの項目（`training_sample_selection_policy`・`most_recent_stored_sample_count`・`stored_sample_recency_decay_factor`）と、その検査を持たない。共有の設定の検査は、`int | None`・`float | None`を、今の前（未対応の型）の扱いへ戻す。
2. The batch sampling shall 選び方の引数を持たず、全部の標本から一様に抜き出す（乱数の使い方と値は今の既定と同じ）。
3. The sweep command shall `--fedsda-training-sample-selections`を持たず、一覧の表は列`training_sample_selection`を持たない。帰属の保留の容量の軸は、今のまま動く。
4. The sweep record shall 計画の項目が減るので、掃引の記録の形式の版を6へ上げる。

### Requirement 2: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存のgoldenと照合が値を変えずに通ること, so that 既定の動作が変わっていないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenと、実旧との照合を、値を変えずに通す（goldenのhelperの「後に足された項目」から、外した3項目の登録を除く）。
2. FedSDAのrunの識別子は、run設定の保存の形が変わるので、変わる（前の結果の出力先へ実行すると、前のrunは計画に入らない）。

### Requirement 3: 文書
**Objective:** As a 研究者, I want 文書が、外したことと、その理由・戻し方を説明すること, so that 後で経緯を追える。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書から抜き出し方の節と引数を外し、研究バックログの7に、外したこと・理由・実装の在りか（commit）を書く。
