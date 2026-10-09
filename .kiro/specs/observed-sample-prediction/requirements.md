# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装は、観測した標本1件に対する学習側の処理（候補検証の進行→損失の監視→保留→警報の処理、または帰属の確定と学習）を`process_observed_sample`の1回の呼出しで行える。予測は未接続で、旧`federated_drift_experiment/clients/fedsda.py`の`_record_prediction`（1590行〜）に当たる処理がない。予測の部品（クラス確率の結合、Fixed-Shareの予測重み、AdaHedgeの診断証拠）は移植済みだが、どこからも呼ばれていない。

変えたいこと: 最終構成（共有特徴抽出部＋非線形残差アダプタ、Fixed-Shareによる重み付き予測、混合予測を常時有効）で、標本1件の予測（保有する全モデルの確率を予測重みで結合）、標本ごとの予測の記録、ラベル観測後の予測重みと診断証拠の更新を、旧と同じ順序と数値で行い、標本1件の処理の、旧と同じ位置（先頭）へつなぐ。

調査で確かめたこと（固定旧748c3aa）:

- 最終3goldenの設定は`soft_routing_context="switching"`、`soft_routing_activation_policy="always"`、`ROUTING_ACTIVE_SET_POLICY="all"`（既定）、`ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS=False`（既定）。
- 有効化方針が`always`のとき、旧の警報側の通知`_on_drift_alarm`は基底の空のhookを呼んで戻るだけで、`_on_drift_resolution`は何も変えない（再生用の標本は常に空で、`resolve`は`always`で即戻る）。したがって、最終構成では、警報側の通知に移植する処理がない。再開案内のfeature名の案（`sample-prediction-and-alarm-notification`）から、通知を外した名前にした。回復期だけ混合予測を有効にする方針（`drift_recovery`）と、そのときの警報側の通知・再生は移植しない（記録はUNPORTED-003）。
- 旧の「切替の通知」（`_on_local_model_change`、2052行〜）は、globalのAdaHedgeの再始動で、held-adahedge-diagnostic-notificationで移植済み。Fixed-Shareの重みは、学習帰属の変更では初期化しない。
- 旧`_record_prediction`の最終構成の分岐が読む・更新するもの: 保有する全モデルの出力（共有特徴は1回だけ計算）、globalのAdaHedge（`expert_router`）、真の概念別のAdaHedge（`oracle_concept_expert_routers`）、Fixed-Share（`switching_expert_router`）、標本ごとの列（`history_accuracy`、`history_concept`、`history_model_id`、`history_routing_*`）、集計の計数（`routing_diagnostics`、`routing_class_diagnostics`、`routing_switching_diagnostics`、`routing_oracle_concept_diagnostics`）、モデル別の除外寄与の診断（`routing_leave_one_out_diagnostics`）、計算量の計数。

## Introduction

研究者が、観測した標本1件に対する予測（ラベルを見る前の重み付き予測、標本ごとの記録、ラベルを見た後の予測重みと診断証拠の更新）が、旧の最終構成と同じ順序と数値で進むことを確かめられるよう、移植済みの予測の部品をつなぎ、標本1件の処理の先頭へ接続する。

## Boundary Context

- **In scope**: 最終構成（Fixed-Shareによる重み付き予測、混合予測を常時有効、保有する全モデルを毎回評価）での、標本1件の予測、標本ごとの予測の記録とその保持、ラベル観測後のFixed-Shareの予測重み・globalの診断証拠・真の概念別の診断証拠の更新、標本1件の処理への接続。
- **Out of scope**: 回復期だけ混合予測を有効にする方針と、そのときの警報側の通知・重みの再生（当面移植しない。UNPORTED-003）。現行モデルだけで予測する経路、評価するモデルを絞る方式、クラス別の文脈・上位の結合（Meta）による予測方式（最終構成で使わない）。モデル別の除外寄与の診断（旧`RoutingLeaveOneOutDiagnostics`。後続のspec）。計算量と所要時間の記録。集約後の予測重みの再較正（サーバ同期のspec）。記録の保存（NPZ・CSV）と集計値の出力。新client・全体run。
- **Adjacent expectations**: クラス確率の計算、Fixed-Shareの予測重み、AdaHedgeの診断証拠は、移植済みの部品をそのまま使い、中の処理を変えない。保有モデルの集合が変わったときの重みと証拠の初期化、学習帰属の変更でのglobalの診断証拠の再始動は、それらの部品と既存の通知が行う（Fixed-Shareの重みは、学習帰属の変更では初期化されない）。標本は、特徴が1行の2次元、ラベルが1行1列で渡される（標本1件の処理の契約）。予測重みのownerと記録のownerを作って渡すのは、新clientの組立ての役目とする。

## Requirements

### Requirement 1: ラベル観測前の重み付き予測

**Objective:** As a 研究者, I want 標本1件について、保有する全モデルの確率をFixed-Shareの予測重みで結合した予測を得たい, so that 旧の最終構成と同じ予測になることを確かめられる

#### Acceptance Criteria

1. When 標本1件の予測を求められたとき, the Observed Sample Prediction shall その時点で保有している全モデル（一時IDのモデルを含む）について、その標本のクラス確率を求める。共有特徴の計算は、標本1件につき1回だけ行う。
2. When 全モデルのクラス確率を求めたとき, the Observed Sample Prediction shall Fixed-Shareの観測前の予測重みで全モデルの確率を重み付けて結合し、結合した確率から予測クラスを決める。
3. The Observed Sample Prediction shall モデル別の確率、予測重み、結合した確率、予測クラスを、その標本の観測ラベルと真の概念IDのどちらにも依存させない。
4. When 予測重みが最大のモデルを決めるとき, the Observed Sample Prediction shall 重みが最大のモデルを選び、同率のモデルが複数あれば現在の学習帰属のモデルを、それが同率に含まれなければIDが最小のモデルを選ぶ。
5. The Observed Sample Prediction shall 予測の間、どのモデルのパラメータも学習状態（訓練・評価の別）も変えず、乱数を消費しない。

### Requirement 2: ラベル観測後の予測重みと診断証拠の更新

**Objective:** As a 研究者, I want ラベルを見た後に、予測重みと診断証拠が、予測に使ったのと同じ確率から求めた損失で更新されてほしい, so that 次の標本の予測と保存する診断が旧と同じになる

#### Acceptance Criteria

1. When 標本の予測と記録を終えたとき, the Observed Sample Prediction shall 予測に使ったのと同じモデル別の確率と観測ラベルから、各モデルの有界損失を求め、Fixed-Shareの予測重みを、予測に使った重みと全モデルの損失で更新する。
2. When 標本の予測と記録を終えたとき, the Observed Sample Prediction shall globalの診断証拠を、その観測前の診断重みと、同じ全モデルの損失で更新する。
3. When 標本に真の概念IDがあるとき, the Observed Sample Prediction shall その概念の診断証拠を、その観測前の診断重みと、同じ全モデルの損失で更新する。
4. If 標本に真の概念IDがないとき, the Observed Sample Prediction shall 概念別の診断証拠を作らず、更新もしない。
5. The Observed Sample Prediction shall 重みと証拠の更新を、予測クラスの決定と記録の後に行う。

### Requirement 3: 標本ごとの予測の記録

**Objective:** As a 研究者, I want 標本ごとの予測の結果と診断の項目が、観測順に保持されてほしい, so that 旧の標本ごとの列と集計の計数を、記録から導ける

#### Acceptance Criteria

1. When 標本1件を予測したとき, the Observed Sample Prediction shall 次の項目を持つ記録を1件作り、記録のownerへ足す: 標本の位置、真の概念ID（なければ「なし」）、観測クラス、結合した予測が正しいか、予測重みが最大のモデルのIDとその重み、実効モデル数（予測重みの二乗和の逆数）、結合した予測またはいずれかのモデルの予測が正しいか、予測重みが最大のモデルの予測が正しいか、globalの診断重みで結合した予測が正しいか、真の概念別の診断重みで結合した予測が正しいか（真の概念IDがなければ「なし」）、確信度が最大のモデルの予測が正しいか。
2. When 確信度が最大のモデルを決めるとき, the Observed Sample Prediction shall 確信度（2値では確率と0.5の差の絶対値、多クラスでは最大のクラス確率）が最大のモデルを選び、同率のモデルが複数あれば、予測重みが大きいもの、次に現在の学習帰属のモデル、次にIDが小さいものを選ぶ。
3. The Sample Prediction Record Store shall 記録を観測順に保持し、保持している記録の、後から変えられない写しを返す。
4. If 足す記録が決まった型でないとき、または位置が保持している最後の記録の位置の次でないとき, the Sample Prediction Record Store shall 何も変えずに拒否する。
5. The Observed Sample Prediction shall 旧の最終構成の標本ごとの列（結合した予測の正否、真の概念ID、予測重みが最大のモデルのID、最大の重み、実効モデル数、いずれかが正しいか、最大重みのモデルが正しいか、真の概念別の診断の正否、混合予測が有効か）と、集計の計数（全体、観測クラス別、Fixed-Share、真の概念別、混合予測を行った標本数）を、記録だけから導ける内容にする。

### Requirement 4: 標本1件の処理への接続

**Objective:** As a 研究者, I want 標本1件の処理が、旧と同じく、予測を最初に行ってほしい, so that 学習側の処理を1回呼ぶだけで、旧の標本処理の全体（範囲外とした、計算量と所要時間の記録、モデル別の除外寄与の診断を除く）と同じ結果になる

#### Acceptance Criteria

1. When 標本1件の処理を求められたとき, the Observed Sample Processing shall 入力の検査の後、候補検証の進行より前に、その標本の予測・記録・予測重みと診断証拠の更新を行う。
2. The Observed Sample Processing shall 予測の結果を、処理の結果に含めて返す。
3. When 同じ初期状態と同じ標本列を与えたとき, the Observed Sample Processing shall 実旧の最終構成のclientの標本処理（予測を含む）と、標本ごとに、予測の記録・Fixed-Shareの予測重み・globalと真の概念別の診断証拠・既存の全状態（保有モデル、学習データ、計数、損失統計、保留、監視、適応記録、警報の記録）・乱数の状態が一致する。

### Requirement 5: 拒否と途中の失敗

**Objective:** As a 研究者, I want 不正な入力が、状態を変える前に拒否されてほしい, so that 拒否の後も、正しい標本から処理を続けられる

#### Acceptance Criteria

1. If 予測が受け取るownerの型、標本の型と形（特徴が1行の2次元、ラベルが1行1列）、位置（非負の整数で、記録の最後の位置の次）、真の概念ID（整数または「なし」）のいずれかが不正であるとき, the Observed Sample Prediction shall どの状態も更新する前に拒否する。
2. If 保有モデルが1つもないとき、または現在の学習帰属のモデルを保有していないとき, the Observed Sample Prediction shall どの状態も更新する前に拒否する。
3. If 標本の中身（特徴の数、ラベルの範囲、有限の値）またはモデルの出力が不正であるとき, the Observed Sample Prediction shall どの状態も更新する前に拒否する。
4. If 標本1件の処理の中で予測が拒否したとき, the Observed Sample Processing shall 後の段へ進まず、どの状態も変えない。
5. If 予測の中で、最初の状態更新より後の段が失敗したとき, the Observed Sample Prediction shall 後の段へ進まない。済んだ段の巻戻しと自動の再試行は行わない。
6. The Observed Sample Prediction shall 並行した呼出しでの結果を保証しない。

### Requirement 6: 新実装だけで動くことと依存の向き

**Objective:** As a 研究者, I want 予測を含む標本1件の処理が、旧実装なしで動いてほしい, so that 新clientの組立てへ進める

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、予測を含む標本1件の処理を、保有モデルが複数ある状態で続けて実行できる。
2. The Observed Sample Prediction shall 既存の予測の部品（クラス確率の計算、Fixed-Shareの予測重み、診断証拠）の中の処理を変えず、新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
