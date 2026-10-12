# Requirements Document

## Introduction
FedSDAのclientで、警報の推定変化区間が最小件数より短いとき、現行モデルへ取り込んだ前区間の標本が、保留に残り、後で、もう一度取り込まれる不具合（[LEGACY-002](../../../docs/research/implementation-findings/legacy-002-retained-assigned-samples-after-short-alarm.md)）を、新実装で直す。2026-10-13にユーザーが採用を決めた（2番目）。同日の計測（提案構成、client 10、標本2500、8 run）では、sine2で、取込み全体の約6〜8%が2回目の取込みだった（短い警報が65回・79回。同じ区間で繰り返す）。ほかのdatasetでは0〜7件。

旧実装（固定748c3aa）との照合を、旧実装を外すまで続けられるように、旧と同じ扱いも設定で選べる形にする（LEGACY-016の修正と同じ方針。分岐が、保留の機能と、警報応答の完了の1か所に閉じるので、この形を使う）。

## Boundary Context
- 対象: 新FedSDAのclientの、短い警報のあとの保留の消費、その設定、基準版のgolden、修正の影響の計測。
- 対象外: 固定旧748c3aaと旧golden、FedDrift・FedAvg（保留の部品を使わない）、短い警報のあとの検出器の扱い（証拠を破棄する。IMPROVE-006。この修正の結果を見てから、別に判断する）、最小件数の値、ほかのLEGACY、旧実装の削除。
- 用語: 「保留」は、学習帰属が未確定の標本（clientの、帰属確定を保留するFIFO）。「前区間」は、警報のとき、保留のうち、推定変化区間より前の標本（現行モデルへ取り込み、評価標本へ保存する）。「変化区間」は、保留の末尾の、推定変化区間の標本。「短い警報」は、変化区間が最小件数（`minimum_change_interval_sample_count`）より少ない警報（適応の結果`alarm_change_interval_too_short`）。

## Requirements
### Requirement 1: 短い警報のあと、取り込んだ前区間を、保留から外す
**Objective:** As a 研究者, I want 同じ標本が2回取り込まれないこと, so that 学習標本・損失統計・評価標本が、観測した標本を1回ずつ反映する。
#### Acceptance Criteria
1. When 短い警報が起き、前区間が1件以上ある, the FedSDA system shall 前区間の標本を保留から外し、変化区間の標本を、順を保って保留に残す。
2. When 短い警報が起き、前区間が無い（保留の全部が変化区間）, the FedSDA system shall 保留を変えない。
3. The FedSDA system shall 短い警報のあとも、これまでと同じく、損失の監視を、現行モデルの基準で再開し、モデルの再利用・候補検証を行わない。
4. The FedSDA system shall 短い警報の完了の記録に、保留から外した標本の位置（前区間の位置）を持たせる。
5. While 既定の扱いで実行している, the FedSDA system shall どの標本も、学習標本として、高々1回だけ取り込む（1つのrunの終わりに、clientの学習標本に、同じ標本が2つ無い）。
6. The FedSDA system shall 短い警報でない警報（保留を全部消費する）と、警報のない標本での確定（容量を超えた最古の標本の確定）の手順を、変えない。
7. The FedSDA system shall 保留の標本を読む処理（警報の推定変化点の算出——保留の件数で打ち切る——と、ADWINで監視する構成の補助の判定——保留の末尾の窓を読む——）が、前区間を外した後の保留（まだ、どのモデルにも取り込まれていない標本だけ）を読むようにする。処理そのものは、変えない。

### Requirement 2: 旧と同じ扱いを、設定で選べる
**Objective:** As a 開発者, I want 旧実装と同じ保留の扱いを選べること, so that 旧実装を外すまで、新旧の照合と、移植し忘れの追加の検証を続けられる。
#### Acceptance Criteria
1. Where 旧と同じ扱いを選んだ, the FedSDA system shall 短い警報のあと、保留を全部残す（旧と同じ結果）。
2. The FedSDA system shall この扱いを、保留の設定（`TrainingDataAssignmentSettings`）の1つの値として持ち、保存するrun設定へ含め、宣言した値以外を拒否する。実験のコマンドの引数には出さない。
3. The FedSDA system shall 最終構成と、その派生の構成（FedSDAの11構成）の既定を、要求1の扱いにする。

### Requirement 3: 基準と照合を、修正の前後で分けて保つ
**Objective:** As a 開発者, I want 修正前の基準が通り続けること, so that 意図した差と、意図しない差を、区別できる。
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、旧と同じ扱いの設定で、変わらず成功させ、照合の入口で、新のclientが旧と同じ扱いであることを確かめる。
2. The validation shall 基準版1・2のgoldenを、それぞれの版の動作に当たる設定（この修正については、旧と同じ扱い）で、値とファイルを変えずに照合し続ける。
3. The validation shall 既定の扱いでの値を、基準版3のgoldenとして、Windowsの基準環境とWSLの両方で足す。基準版2の扱いの実行で、前区間のある短い警報（2つの扱いの状態が、初めて分かれる事象）が1回も起きないケースは、基準版3の値（指標・表・パラメータ）が、基準版2の値と同じであることを確かめる。その事象が起きるケースについては、名前を挙げた2つのケース（`fedsda-sine2-events`・`fedsda-sine2-reference-reuse`）で、基準版2の扱いの実行に、同じ標本の2回目の取込みがあり、基準版3の値が基準版2と違うことを確かめる（事象が起きても、値が違うとは限らない——例: runの最後の標本での短い警報——ので、一般には主張しない）。名前を挙げたケースで値が違わなかった場合は、要求を緩めず、その事実を記録して、ユーザーへ報告する。
4. The validation shall 基準版3の、FedSDAの全ケースで、runの終わりに、同じ標本の2回目の取込みが0であることを確かめる（FedDrift・FedAvgのケースは、保留の部品を持たないので、対象外）。
5. If 基準版3の扱いで、あるケースが、基準版2で必須としていた重要な経路（候補の採用・棄却、別の保有モデルの再利用、統合、区間ごとの不合格など）を通らなくなった, the validation shall その経路を通る条件のケースを、基準版3へ足す（基準版2までのケースと設定は、変えない）。

### Requirement 4: 修正の影響を、測って報告する
**Objective:** As a 研究者, I want 修正で精度や通信量がどう変わるかを知ること, so that 論文の実験をやり直す範囲を判断できる。
#### Acceptance Criteria
1. The validation shall LEGACY-016の計測と同じ8条件（提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0・1）で、旧と同じ扱いと既定の扱いを実行し（登録の扱いは、どちらも既定）、精度・定常精度・通信量・最終のグローバルモデル数・短い警報の数・同じ標本の2回目の取込みの数・学習標本の総数・変化点の推定の誤差を比べて、記録する。保留の読み方が変わる（要求1.7）ADWINで監視する2構成も、sine2のseed 0・1で、同じ比較を行う。
2. If 既定の扱いで精度などが下がる条件がある, the report shall その条件と差を、ユーザーへ伝える。
