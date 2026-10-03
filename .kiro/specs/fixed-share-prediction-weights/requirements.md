# 要求: Fixed-Share予測重み

## 導入と範囲

研究者が最終FedSDAの予測重み更新を、モデル推論・学習・割当から独立して検証できるようにする。
対象は旧748c3aaのSwitchingExpertRouterに対応する予測重みの更新と損失再生。
モデル出力のtensor混合、監視・候補採否、FIFO損失の再計算、サーバ同期、研究指標と完全なFedSDA実行は後続範囲。
Fixed-Shareは既存手法の名称として使用する。新しい公開名と役割はnaming.mdでレビューする。

## 要求

### Requirement 1: 条件と状態の独立性

**目的:** 研究者が条件誤指定や他のrunからの状態混入を避ける。

1. When 予測重みの管理を開始する, the Prediction Weight System shall 固定条件を明示的に受け取り、独立した状態を作る。
2. If 時間尺度が2未満または入力の型・値が不正である, the Prediction Weight System shall 理由を示して拒否し、重み・累積分散・全計数を含む所有状態を変更しない。
3. The Prediction Weight System shall 呼出元の入力とグローバル設定・乱数を変更せず、返した観測値の変更から所有状態を隔離する。

### Requirement 2: モデル集合と予測前の重み

**目的:** 研究者が保持モデル集合の変更と予測用重みを追跡できる。

1. When 初めて非空のモデル集合を受け取る, the Prediction Weight System shall 全モデルを一様な予測重みで開始する。
2. When 保持モデルのID集合が変わる, the Prediction Weight System shall 一様重みとゼロの累積分散へ再構成し、初回を除く集合変更を記録する。
3. When 同じ集合を異なる列挙順で受け取る, the Prediction Weight System shall ID昇順で同じ重みを返し、再構成しない。学習割当先だけが変わっても、モデルID集合が同じなら重み・累積分散を保持する。
4. If モデル集合が空または重複IDを含む, the Prediction Weight System shall 拒否する。整数の負IDは一時モデルとして受理する。

### Requirement 3: ラベル観測後の更新

**目的:** 研究者が現在のラベルを予測に混入せず、次回用の重みを更新できる。

1. When モデルごとの観測損失と予測時の重みを受け取る, the Prediction Weight System shall 次回用重みと累積分散を旧基準と同じ数値・演算順で更新する。
2. The Prediction Weight System shall 有限な観測損失を0から1へ制限して更新に使い、時間尺度に対応する一様分布への重み再配分を維持する。
3. While モデルが一つである, the Prediction Weight System shall 重み1を維持し、分散更新やleader切替を追加しない。
4. If 損失と予測重みのID集合が一致しない、または重み・損失が不正である, the Prediction Weight System shall 更新前に拒否する。

### Requirement 4: leaderと観測記録

**目的:** 研究者が重み最大モデルの変化を学習データ割当と区別して追跡できる。

1. When 最大重みが同率である, the Prediction Weight System shall 指定された優先IDが同率候補に含まれる場合はそれを返し、それ以外は最小IDを返す。
2. When 更新による最大重みモデルの変更を記録する, the Prediction Weight System shall 最小IDによる同率解決を使って旧基準と同じ切替件数を記録する。
3. The Prediction Weight System shall leaderを予測の観測事実として返し、学習データの割当先を変更しない。

### Requirement 5: 再生と集約後再較正

**目的:** 研究者が集約後のモデル損失を時系列順に再評価できる。

1. When 非空の損失列を再生する, the Prediction Weight System shall 重み・累積分散を消去してから列順で予測重み取得と損失更新を再現する。
2. When 集約後の非空損失列を再生する, the Prediction Weight System shall 再較正回数と再生標本数を記録して再生する。
3. When 空の損失列を受け取る, the Prediction Weight System shall 状態と再較正件数を変更しない。
4. When 集約後の明示的な初期化を受け取る, the Prediction Weight System shall 重み・分散を消去して再較正回数を一つ増やす。
5. The Prediction Weight System shall 損失の再計算やモデル学習を行わず、入力された損失の順序を維持する。

### Requirement 6: 基準照合と境界

**目的:** 研究者が構造変更による数値差を局所化できる。

1. The Prediction Weight System shall 一様初期化、最良モデル変化、集合変更、負ID、同率、損失制限、単一モデル、再生、空列の挙動を旧基準と照合できる。
2. The Prediction Weight System shall 旧実装への依存・互換名・真の概念入力を新しい実行経路へ導入しない。
3. The Prediction Weight System shall この部分的な数値照合を、最終FedSDA全体の新経路golden一致として扱わない。
