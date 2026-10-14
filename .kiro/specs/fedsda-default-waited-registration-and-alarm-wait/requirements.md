# Requirements Document

## Introduction
LEGACY-016・LEGACY-002・IMPROVE-006の本規模の比較（2026-10-14、研究室サーバ、8通り×120 run）と、FedDriftとの比較の結果、ユーザーは、FedSDAの既定を「案Y」にすると決めた（2026-10-14）。案Yは、次の3つの組合せ。

- 登録: 送信保留のモデルを、現行モデルとして待った後に登録する（`register_pending_model_after_waiting_as_current_model`。実装済み。[spec](../pending-model-registration-after-waiting-as-current-model/design.md)）。
- 短い警報のあとの保留: 取り込んだ前区間を保留から外す（`release_earlier_samples_assigned_to_current_model`。既に既定）。
- 警報: 推定変化区間が最小件数に達するまで、警報を待つ（`wait_until_change_interval_reaches_minimum`。実装済み。[spec](../alarm-wait-for-minimum-change-interval/design.md)。IMPROVE-006）。

判断の根拠と数値は、[LEGACY-016の記録](../../../docs/research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)の「本規模の比較（待つ登録と待つ警報を含む8通り）と、案Yの判断」。どちらの扱いも、実装と照合は済んでいて、このspecは、既定の値と、それに伴う構成・基準版・文書の整合だけを扱う。

## Boundary Context
- 対象: 最終構成の既定（登録と警報の2つの値）、FedSDAの派生の構成の設定、基準版のgolden（版4の設定の明示と、版5の追加）、既定に依るtest、文書。
- 対象外: 2つの扱いの動作そのもの（変えない）、FedDrift・FedAvg、固定旧と旧golden、比較scriptの水準（扱いを明示して組み立てるので、変えない）、旧実装の削除。

## Requirements
### Requirement 1: FedSDAの既定を案Yにする
**Objective:** As a 研究者, I want 既定が案Yであること, so that 通常の実験のコマンドが、決めた扱いで実行される。
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定の登録の扱いを`register_pending_model_after_waiting_as_current_model`、警報の扱い（推定変化区間が短いときの扱い）を`wait_until_change_interval_reaches_minimum`にする（短い警報のあとの保留の扱いは、既定の`release_earlier_samples_assigned_to_current_model`のまま）。
2. The FedSDA system shall e-SRの検出器を使う構成（提案構成と、ADWINの2構成を除く8つの派生の構成。計9構成）で、既定の2つの値を引き継ぐ。
3. The FedSDA system shall ADWINの検出器を使う2つの派生の構成（`fedsda-classwise-adwin-detection`・`fedsda-overall-loss-adwin-detection`）では、警報の扱いを`raise_alarm_immediately`に明示する（ADWINと待つ扱いの組合せは、設定の検査で拒否される。登録の扱いは、既定を引き継ぐ）。
4. The FedSDA system shall 実験のコマンドの引数を変えない（掃引のコマンドの引数の一覧を、構成の既定のtestで確かめる）。

### Requirement 2: これまでの基準と照合を保つ
**Objective:** As a 開発者, I want これまでの基準が値を変えずに通ること, so that 既定の変更による差と、意図しない差を区別できる。
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、変わらず成功させる（実旧と比べる設定は、旧と同じ扱いを明示している）。
2. The validation shall 基準版1〜4のgoldenを、それぞれの版の扱いで、値を変えずに照合し続ける（版4は、警報の扱いを`raise_alarm_immediately`と明示して組み立てる）。

### Requirement 3: 新しい既定を基準版5として足す
**Objective:** As a 開発者, I want 新しい既定の値が基準として固定されること, so that 以後の変更で、既定の動作が変わったときに気づける。
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版5のgoldenとして、Windowsの基準環境とWSLの両方で足す（研究室サーバ用は対象外）。
2. The validation shall 基準版5で警報を待つ扱いを使うケース（e-SRの検出器のFedSDAのケース）ごとに、基準版4の扱いの実行で、待つ扱いなら警報を待つ場面（警報が上がり、保留の件数と推定変化区間の小さい方が、最小件数より短い。候補検証を保持している間を含む）が起きた回数を数え、0のケースでは、基準版5の値（指標・表・パラメータ）が基準版4と同じであること、1以上のケースでは差があることを確かめる。1以上のケースが、少なくとも1つあること（差が出ることは、実行で確かめる観測）。
3. The validation shall ADWINの2構成のケースと、FedDrift・FedAvgのケースは、基準版4と基準版5で、同じ値であることを確かめる（警報の扱いが2つの版で同じなので、事象は定義により0とする）。

### Requirement 4: 文書を合わせる
**Objective:** As a 研究者, I want 文書が新しい既定を正しく説明すること, so that 実験の条件を取り違えない。
#### Acceptance Criteria
1. The documentation shall （実装レビューで文書の差分を確かめる）実験の入口の文書、基準の文書、IMPROVE-006の記録と一覧、LEGACY-016の記録、再開案内、比較scriptの説明で、新しい既定と、その根拠への参照を書く。
