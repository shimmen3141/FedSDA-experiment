# Requirements Document

## Project Description (Input)
比較実験の利用者は、新実装の単一FedDrift runを実行・保存できるが、新しい掃引CLIからは選べない。FedSDAと同じ入口からFedDriftの条件を展開し、実行・保存・再開・一覧化できるようにする。

## Boundary Context
対応datasetは既存新実装のsine2・sea2・sea4・circle2・mnist2・mnist4の6つ。blobsは既存新実装の対象外を維持する。対象は既定のFedDrift構成、検出バッチと損失増分閾値の掃引、既存CLIへの接続。指標・学習・時刻進行・成果物形式は既存specの契約を使う。全構成goldenの新規作成、旧CLI互換、追加診断、描画、任意設定のCLI公開は対象外。

## Requirements

### Requirement 1: 既定構成
**Objective:** 比較実験の利用者が、規模・dataset・seedと検出粒度を指定して、旧の既定条件に対応するFedDriftを実行できる。
#### Acceptance Criteria
1. When 利用者がfeddriftと実行条件を指定する, the sweep command shall 旧のFedDriftの既定に対応する完全条件を作る。
2. The sweep command shall 合成datasetとMNISTのモデル幅・学習率を、それぞれ旧のdataset別の既定に合わせる。
3. If 不正な実行条件や閾値が指定される, the sweep command shall 実験の生成・実行・出力先作成の前に拒否する。

### Requirement 2: 条件展開
**Objective:** 比較実験の利用者が、手法に適用される軸だけを展開できる。
#### Acceptance Criteria
1. When FedDriftを含む掃引を計画する, the sweep command shall 集約間隔の各値を検出バッチ件数にも適用し、指定された損失増分閾値の各値と組み合わせる。閾値省略時は0.1だけを使う。
2. When FedSDAとFedDriftを同じ掃引へ指定する, the sweep command shall FedSDAを閾値で複製せず、構成・dataset・間隔・seed・FedDrift閾値の順で計画する。
3. If 閾値軸が空・重複・非有限・負数である、またはFedDriftを含まない掃引で明示される, the sweep command shall 出力を作る前に拒否する。
4. The sweep command shall 完全な条件と構成名から識別子を決め、閾値変更を異なるrunとして扱い、既存FedSDAの識別子を維持する。

### Requirement 3: 実行・再開・一覧
**Objective:** 比較実験の利用者が、FedDriftを保存・再開し、共通指標を比較できる。
#### Acceptance Criteria
1. When 計画を実行する, the sweep command shall 各手法の新実装で実行・保存し、構成ごとに結果を分け、逐次と複数workerで完全設定・全指標・全記録表の値を一致させる。実行環境の記録・経過時間・掃引開始終了時刻・保存先・実行/skip状態は一致対象に含めない。
2. When 同じ条件の読める成果物がある, the sweep command shall 再実行せずに飛ばす。If 片方だけ・破損・手法や条件の不一致がある, the sweep command shall 上書きせず失敗として扱う。
3. When 結果を一覧化する, the sweep command shall FedDriftの検出バッチと閾値をCSVから判別できるようにし、全指標と掃引条件・結果・実行環境の記録を保存する。

### Requirement 4: コマンドの統合
**Objective:** 比較実験の利用者が、既存入口で新しい手法も扱える。
#### Acceptance Criteria
1. When 利用者がhelpまたは計画表示を行う, the sweep command shall FedDriftと専用閾値軸の使い方を表示し、計画表示ではファイルやディレクトリを作らない。
2. The sweep command shall 既存の終了コード0/1/2、MNIST事前確認、workerと再開の規約を維持する。
3. The sweep command shall 旧実装をimportせずに計画・実行・保存・読込みを完了し、既存FedSDAの条件・値・保存とgoldenを維持する。
