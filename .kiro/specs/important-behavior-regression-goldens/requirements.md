# Requirements Document

## Project Description (Input)
研究者は移植済みのFedSDA構成・FedDrift・FedAvgを比較する。現在は実旧対照と一部の環境別goldenを持つが、残るLEGACY/IMPROVE適用前の新実装の重要動作を固定する基準が不足する。新実装だけで照合できる代表goldenを追加し、基準版と数値環境を区別して管理する。対象は代表条件であり全条件の直積ではない。旧実装・旧golden、手法の挙動、成果物の形式、既知問題の採否を変更しない。

## Requirements
### Requirement 1: 新実装の重要動作の回帰
**Objective:** 研究者として、移植済み構成の観測結果を固定し、後続変更の意図しない影響を検出したい。
#### Acceptance Criteria
1. The 回帰検証 shall 移植済み13構成すべてについて小さい代表runの完全設定・全指標・全記録列・最終グローバルパラメータを固定して照合する。
2. The 回帰検証 shall 追加のFedSDA sine2 runで候補採用・前半区間での棄却・別モデル再利用・学習帰属切替・モデル統合・予測重み再較正を、mnist2 runで候補採用・後半区間での棄却・学習帰属切替・予測重み再較正を実際に通ったことをケースごとに検査する。
3. The 回帰検証 shall FedDriftの追加代表runでドリフト検出・モデル登録・複数ローカルラウンド・観測済みバッファへの終端処理と生成末尾の未処理を、FedAvgの代表runで同期と未処理末尾、および完了intervalなしを検査する。
4. When 基準と異なる結果が出た場合, the 回帰検証 shall ケースと相違した指標名・記録列名・パラメータ名（設定はfield path）を示して失敗し、基準を変更しない。

### Requirement 2: 基準版と生成の来歴
**Objective:** 研究者として、意図したアルゴリズム変更と環境差を区別したい。
#### Acceptance Criteria
1. The golden shall 動作の基準版・記録形式の版・生成元commit・完全設定・数値計算環境を区別して保存する。
2. When 新しい基準を生成する場合, the 生成処理 shall 独立processで同じ条件を2回実行した結果の一致を確認してから出力する。
3. If 出力先が既に存在する場合, then the 生成処理 shall 上書きせず失敗する。
4. The 回帰検証 shall 明示した現在の基準版を使用し、未知の版や壊れた基準を失敗として扱う。

### Requirement 3: 数値環境と検証入口
**Objective:** 開発者として、数値環境が違うために既存基準を更新せず、検証範囲を把握したい。
#### Acceptance Criteria
1. The 回帰検証 shall Windows固定venvとWSL Ubuntuのそれぞれについて対応する環境の新goldenを検証する。
2. If 実行環境に対応するgoldenがない場合, then the 回帰検証 shall 数値照合を理由つきでskipし、基準の構造・ケース定義の検証は実行する。
3. The CI shall 新goldenの構造検査を必須検査に含め、ホストの数値照合を既存方針に合わせて診断として実行する。
4. The 検証処理 shall 新実装の実行と基準の照合に旧実装や旧testを必要としない。
