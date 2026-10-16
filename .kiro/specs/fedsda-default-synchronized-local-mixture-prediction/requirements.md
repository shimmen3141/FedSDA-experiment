# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: 本規模の比較（2026-10-15〜16、研究室サーバ、研究バックログの8）で、予測を「同期の時点のモデルの結合予測と、局所更新したモデルの結合予測を、clientごとの重みで混ぜる」方式（設定の値`mixture_of_synchronized_and_local_parameters`）にし、Fixed-Shareの時間尺度Tを120にした構成が、6データ・4つの集約間隔で、既定（局所の予測・T30）以上の精度だった。T120は、T30〜240の中で、どのデータ・間隔でもT30とほぼ同等以上だった（最小差は−0.03ポイント）。
- 何を変えるか: FedSDAの最終構成の既定を、その構成（混ぜる予測・T120。帰属の保留の容量は30のまま）にする。基準版6を足す。

## Introduction
2026-10-16の会話で、ユーザーは「混ぜる予測・容量30・T120」を既定にすると決めた。混ぜる予測とTの分離は実装済み（[fedsda-synchronized-local-mixture-prediction](../fedsda-synchronized-local-mixture-prediction/tasks.md)、[fedsda-fixed-share-time-scale-axis](../fedsda-fixed-share-time-scale-axis/tasks.md)）。このspecは、既定の値と、それに伴う構成・基準版・照合・文書の整合だけを扱う。

## Boundary Context
- 対象: 最終構成の既定（予測に使うパラメータ、T）、FedSDAの派生の構成（組み合わせられない構成の明示、局所の予測の構成の追加）、旧と比べる設定の入口、基準版のgolden（版1〜5の明示と、版6の追加）、既定に依るtest、文書。
- 対象外: 混ぜる予測とFixed-Shareの動作（変えない）、帰属の保留の容量（30のまま）、FedDrift・FedAvg、固定旧と旧golden、研究室サーバ用のgolden。

## Requirements
### Requirement 1: FedSDAの既定を、混ぜる予測・T120にする
**Objective:** As a 研究者, I want 既定が決めた構成であること, so that 通常の実験のコマンドが、その構成で実行される。
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定の`prediction_model_parameter_source`を`mixture_of_synchronized_and_local_parameters`、`fixed_share_weight_redistribution_time_scale_samples`を120にする（`pending_assignment_buffer_capacity_samples`は30のまま）。
2. The FedSDA system shall 派生の構成で、既定の2つの値を引き継ぐ。ただし、混ぜる予測と組み合わせられない構成（現在のモデルだけで予測する`fedsda-single-model-prediction`）は、`current_local_parameters`を明示する。すでに予測に使うパラメータを明示する構成（`fedsda-synchronized-model-prediction`、`fedsda-synchronized-local-mixture-prediction`）は、その値のまま（Tは既定を引き継ぐ）。
3. The FedSDA system shall 構成`fedsda-local-parameter-prediction`（既定から、予測に使うパラメータだけを`current_local_parameters`にした構成。混ぜる予測の寄与を比べる用）を足す。
4. The FedSDA system shall 全部の構成の組立てが例外を出さず、実験のコマンドの引数を変えない。

### Requirement 2: これまでの基準と照合を保つ
**Objective:** As a 開発者, I want これまでの基準が値を変えずに通ること, so that 既定の変更による差と、意図しない差を区別できる。
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、変わらず成功させる。旧と比べる設定の入口は、予測に使うパラメータを`current_local_parameters`、Tを帰属の保留の容量と同じ値（30。旧の1/N_FIFO）に明示し、照合の入口の検査でそれを確かめる。
2. The validation shall 基準版1〜5のgoldenを、それぞれの版の扱い（予測に使うパラメータ`current_local_parameters`、T30）で、値を変えずに照合し続ける。

### Requirement 3: 新しい既定を基準版6として足す
**Objective:** As a 開発者, I want 新しい既定の値が基準として固定されること, so that 以後の変更で、既定の動作が変わったときに気づける。
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版6のgoldenとして、Windowsの基準環境とWSLの両方で足す（研究室サーバ用は対象外）。
2. The validation shall 基準版5と6の差が予測の側に限られることを、全ケースで確かめる: 集約後のモデルのパラメータ（`global_parameters`）と経路の記録（`coverage`）は版5と同じ。記録の表と指標は、予測の側の項目として設計で定めた一覧（完全な名前）の項目だけが違ってよい。学習の側の計算回数は、版5と同じ。
3. The validation shall 重み付けの予測を使うFedSDAのケースでは、ケースごとに予測の側の項目（予測の記録・予測の指標・推論の回数）のどれかに差があり、予測の記録に差があるケースが少なくとも1つあること、FedDrift・FedAvgと現在のモデルだけで予測するケースでは全部が同じであることを確かめる。

### Requirement 4: 文書を合わせる
**Objective:** As a 研究者, I want 文書が新しい既定を正しく説明すること, so that 実験の条件を取り違えない。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書（構成の表、容量とTの節）、基準の文書（版6）、研究バックログの8（既定の判断）、最終提案構成の文書（固定設定）、再開案内で、新しい既定と、その根拠への参照を書く。
