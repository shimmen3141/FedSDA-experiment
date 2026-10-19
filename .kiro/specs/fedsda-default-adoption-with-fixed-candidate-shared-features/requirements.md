# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: 研究ツール（`tools/research_provisional_candidate_expert/`）の本規模の比較（2026-10-19〜20、uranium、6データ・4つの集約間隔・seed 0〜9、5000標本）で、「候補検証をせずに候補をその場で採用し、候補の学習では共有部を更新しない」構成が、今の既定（候補検証あり）に対して、24条件中18条件で精度が高く、有意の目安で下回る条件は無かった。通信量は、下りが最大で約25%増えた。今の既定の候補検証は、警報の後の検証の期間・警報の先送り・4通りの判定を持ち、論理が複雑。
- 何を変えるか: FedSDAの最終構成の既定を、その構成（候補検証なし＋候補の共有部の固定）にする。これまでの既定（候補検証あり、候補の学習で共有部も更新する）は、構成として選べるように残す。基準版7を足す。

## Introduction
2026-10-20の会話で、ユーザーは「候補検証なし＋候補の共有部の固定」を既定にすると決め、候補検証をオプションとして残すかどうかの判断を主担当に委ねた。主担当は残すと判断した（理由は設計）。結果と判断の記録は[研究記録](../../../docs/research/fedsda-provisional-candidate-expert-20261019.md)。

## Boundary Context
- 対象: 候補の学習で共有部を更新するかどうかの設定（新しい項目）と、その実行、候補検証と共有部の固定の組合せの制約、最終構成の既定、FedSDAの派生の構成（これまでの既定を選ぶ構成の追加、候補検証を外した構成の明示）、旧と比べる設定の入口、基準版のgolden（版1〜6の明示と、版7の追加）、既定に依るtest、候補検証の有無を構成名で選んでいる研究ツールの指定、文書。
- 対象外: 候補検証の処理そのもの（消さない、変えない）、候補の学習の他の部分（初期値、エポック、早期終了）、採用の手順、FedDrift・FedAvg、固定旧と旧golden、研究室サーバ用のgolden。

## Requirements
### Requirement 1: 候補の学習で、共有部を固定できる
**Objective:** As a 研究者, I want 候補の学習で共有部を更新しない選択肢, so that 候補の採用が、既存のモデルの入力（共有部）を書き換えないようにできる。
#### Acceptance Criteria
1. The FedSDA system shall 候補の学習の設定に、共有部を更新するかどうかを選ぶ項目を持つ（値は「候補と一緒に更新する」と「固定する」の2つ）。
2. When 「固定する」が選ばれている, the FedSDA system shall 候補の学習の全部の更新で、候補の共有部のパラメータを変えず、概念固有部（adapterと出力層）だけを更新する。
3. When 「候補と一緒に更新する」が選ばれている, the FedSDA system shall これまでと同じ値・同じ乱数の消費で候補を学習する。
4. If 候補検証をする採否の方針と「固定する」が同時に指定された, the FedSDA system shall run設定の組立てで拒否する。
5. Where 共有部を持たない構造（独立のモデル）, the FedSDA system shall どちらの値でも、これまでと同じ結果にする。

### Requirement 2: FedSDAの既定を、候補検証なし＋候補の共有部の固定にする
**Objective:** As a 研究者, I want 既定が決めた構成であること, so that 通常の実験のコマンドが、その構成で実行される。
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定の候補の採否の方針を「検証せずに採用する」、候補の共有部を「固定する」にする（ほかの既定は変えない）。
2. The FedSDA system shall 構成`fedsda-with-candidate-validation`（これまでの既定: 候補検証あり、候補と一緒に共有部を更新する）を足す。
3. The FedSDA system shall 構成`fedsda-without-candidate-validation`を、これまでと同じ設定（候補検証なし、候補と一緒に共有部を更新する）のまま保つ。
4. The FedSDA system shall ほかの派生の構成で、既定の2つの値を引き継ぐ。全部の構成の組立てが例外を出さず、実験のコマンドの引数を変えない。

### Requirement 3: これまでの基準と照合を保つ
**Objective:** As a 開発者, I want これまでの基準が値を変えずに通ること, so that 既定の変更による差と、意図しない差を区別できる。
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、変わらず成功させる。旧と比べる設定の入口は、候補の共有部を「候補と一緒に更新する」に明示し、照合の入口の検査で確かめる。旧のgoldenの条件は、候補検証ありに明示する。
2. The validation shall 基準版1〜6のgoldenを、それぞれの版の扱い（候補検証あり——候補検証を外した構成のケースを除く——、候補と一緒に共有部を更新する）で、値を変えずに照合し続ける。
3. The validation shall 候補検証の処理（検証の期間、判定、警報の先送り、未完了の検証の締め）の既存のtestを、変わらず成功させる。

### Requirement 4: 新しい既定を基準版7として足す
**Objective:** As a 開発者, I want 新しい既定の値が基準として固定されること, so that 以後の変更で、既定の動作が変わったときに気づける。
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版7のgoldenとして、Windowsの基準環境とWSLの両方で足す（研究室サーバ用は対象外）。
2. The validation shall 版7で、候補検証の経路を守るケース（版6の大きい条件のFedSDAの4ケース）を、構成`fedsda-with-candidate-validation`で実行し、版6と同じ値であることを確かめる。FedDrift・FedAvg・`fedsda-without-candidate-validation`のケースも、版6と同じであることを確かめる。
3. The validation shall 新しい既定の経路（候補を検証せずに採用し、学習帰属を切り替え、統合と予測の重みの再較正が起きる）を、大きい条件の新しいケースで守り、その経路が実際に起きたことを確かめる。

### Requirement 5: 研究ツールと文書を合わせる
**Objective:** As a 研究者, I want 研究ツールの対照の意味と、文書の説明が、既定の変更の後も正しいこと, so that 実験の条件を取り違えない。
#### Acceptance Criteria
1. The research tools shall 候補検証ありの対照を構成名`fedsda`で選んでいる箇所を、`fedsda-with-candidate-validation`へ替え、対照の意味を変えない。
2. The documentation shall 実験の入口の文書（構成の表）、基準の文書（版7）、最終提案構成の文書（固定設定）、研究記録（既定の判断）、再開案内で、新しい既定と、その根拠への参照を書く。
