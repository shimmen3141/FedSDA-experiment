# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: サーバは、登録・集約・配布まで行える（global-model-distribution）。旧の最終構成のサーバは、配布の直後に、全clientの予測の重みと診断証拠を、配布の後のモデルで再較正するが、新にはこの処理がない。このため、新の登録→集約→配布は、旧のサーバの`run_round`そのものとは照合できていない。

変えたいこと: 旧の集約後の再較正（`recalibrate_routing_after_aggregation`の`fifo_replay`）を、旧と同じ順序・数値で行う。

調査で確かめたこと（固定旧748c3aa）:

- サーバ（`servers/shared_backbone.py` 25〜30行）: `run_round`は、登録→集約→（統合）→配布の後、再較正の方式が`none`でなければ、全clientへ、登録順に`recalibrate_routing_after_aggregation`を呼ぶ。
- client（`clients/shared_backbone.py` 80〜126行）の`fifo_replay`: (1)保留中の標本（帰属を確定する前の標本）について、保有する全モデルの損失の列を作る。(2)globalの診断証拠（AdaHedge）を、その列で再生する。(3)真の概念別の診断証拠を、作成済みのもの全部、再始動する。(4)Fixed-Shareの予測の重みを、同じ列で再生する。
- 損失の列（同126〜168行）: 保留中の標本がないか、保有モデルが1つ以下なら、空。そうでなければ、モデルIDの昇順で、各モデルの、標本ごとの有界損失（2値は出力と観測ラベルの差の絶対値、多クラスは1−観測クラスの確率）を、勾配なしで計算し、標本の順に、モデルIDから損失への対応を並べる。
- AdaHedgeの再生（`expert_routing.py` 174〜193行）: 列が空なら何もしない。空でなければ、再較正の回数を1、再生した標本数を列の長さだけ足し、証拠（累積損失とmixability gap）を消してから、列の順に、観測前の重みの取得と、損失の観測による更新を繰り返す。
- AdaHedgeの集約後の再始動（同169〜173行）: 証拠を消し、集約後の再始動の回数と、再較正の回数を、それぞれ1足す（証拠が空でも数える）。
- Fixed-Shareの再生（同511〜518行）は、移植済み（`replay_observed_losses_after_aggregation`）。
- 最終構成で空のままの旧の状態（予測クラス別・shadowのAdaHedge）と、計算量の記録は、移植しない。

## Introduction

研究者が、配布の後の予測の重みと診断証拠の再較正を、旧と同じ結果になる呼出しで行えるようにする。これで、クラスタリングなしのラウンドを、旧のサーバの`run_round`そのものと照合できる。

## Boundary Context

- **In scope**: client 1つの、集約後の再較正（保留中の標本の損失の列の計算、globalの診断証拠の再生、真の概念別の診断証拠の再始動、Fixed-Shareの重みの再生）。そのために要る、単一の診断証拠の操作の追加（集約後の再生、集約後の再始動、3つの計数）。clientの操作。
- **Out of scope**: 再較正の方式の選択（新の設定は、この方式だけを持つ）。クロス評価・クラスタリング・統合。実行の枠のサーバの操作の実体と、全体runの接続（全clientへ順に呼ぶのは、呼出し側）。計算量の記録。
- **Adjacent expectations**: Fixed-Shareの重みの再生、標本ごとの有界損失の計算、診断証拠の観測前の重みの取得と更新は、移植済みの部品をそのまま使い、中の処理を変えない。

## Requirements

### Requirement 1: 損失の列

**Objective:** As a 研究者, I want 再較正に使う損失が、配布の後のモデルで、旧と同じ規則で計算されてほしい, so that 再較正の後の重みと証拠が旧と同じになる

#### Acceptance Criteria

1. When 再較正を求められ、保留中の標本が1件以上あり、保有モデルが2つ以上あるとき, the Recalibration shall 保有する全モデル（一時IDを含む）について、保留中の全標本の有界損失を、勾配なしで計算し、標本の観測順に、モデルIDの昇順の「モデルIDから損失」の対応を並べた列を作る。
2. When 保留中の標本がないか、保有モデルが1つ以下のとき, the Recalibration shall 空の列を使う。
3. The Recalibration shall 損失の計算で、モデルのパラメータ・勾配・optimizerの状態・訓練の別、保留中の標本、乱数を変えない。

### Requirement 2: 診断証拠の再生と再始動

**Objective:** As a 研究者, I want 診断証拠が、旧と同じく再生・再始動されてほしい, so that 診断の指標と計数が旧と同じになる

#### Acceptance Criteria

1. When 損失の列が空でないとき, the Recalibration shall globalの診断証拠の、再較正の回数を1、再生した標本数を列の長さだけ足し、証拠を消してから、列の順に、観測前の診断重みの取得と、損失の観測による更新を行う。
2. When 損失の列が空のとき, the Recalibration shall globalの診断証拠を変えない。
3. When 再較正を求められたとき, the Recalibration shall 作成済みの真の概念別の診断証拠を全部、損失の列が空かどうかによらず、再始動する（証拠を消し、集約後の再始動の回数と、再較正の回数を、それぞれ1足す）。
4. The 診断証拠 shall 集約後の再生と再始動で、概念操作による再始動の回数を変えない。モデル集合の変化の回数は、再生の列の中で、行のモデル集合が前の行と違うときだけ、通常の観測と同じ規則で増える（証拠を消した直後の最初の行では増えない。旧と同じ。再較正が作る列は、全行が同じモデル集合なので、増えない）。

### Requirement 3: 予測の重みの再生

**Objective:** As a 研究者, I want Fixed-Shareの予測の重みが、同じ損失の列で再生されてほしい, so that 配布の後の予測が旧と同じ重みから始まる

#### Acceptance Criteria

1. When 再較正を求められたとき, the Recalibration shall globalの診断証拠の再生、真の概念別の診断証拠の再始動の後に、Fixed-Shareの予測の重みを、同じ損失の列で再生する（列が空なら、重みも計数も変えない）。
2. The Recalibration shall 保有モデル、損失統計、学習データ、評価標本、計数、現在の学習帰属、保留、監視、候補検証、送信保留、適応記録、予測の記録を変えない。

### Requirement 4: 実旧との一致

**Objective:** As a 研究者, I want クラスタリングなしのラウンドが、旧のサーバの`run_round`と同じ結果になってほしい, so that 統合を除く全体の流れを、旧と同じ状態で進められる

#### Acceptance Criteria

1. When 同じ初期モデル・統計・設定・標本列・乱数を与えたとき, the 再較正 shall 実旧のサーバの`run_round`（クラスタリングなし、再較正の方式は`fifo_replay`）と、実`__init__`で作った実旧のclient複数に対して、ラウンドごとに、標本処理→保留中の学習→登録→集約→配布→再較正→送信待ちの進行、を行い、サーバと各clientの全状態（診断証拠の3つの計数を含む）と乱数の状態が一致する。
2. The 対照 shall 損失の列が空でない再較正（一時IDのモデルを含む場合を含む）、保有モデルが1つで空になる再較正、真の概念別の診断証拠の再始動、2値と多クラスを通る。
3. When 単一の診断証拠へ、集約後の再生と再始動を行ったとき, the 診断証拠 shall 実旧のAdaHedgeの同じ操作と、累積損失・mixability gap・全計数が一致する。

### Requirement 5: 拒否

**Objective:** As a 研究者, I want 不正な入力が、状態を変える前に拒否されてほしい, so that 重みと証拠が、途中まで更新されたまま残らない

#### Acceptance Criteria

1. If ownerが決まった型でないとき, the Recalibration shall どの状態も変える前に拒否する。
2. If 診断証拠の再生へ渡す損失の列に、不正な行（Mappingでない、IDがbuiltin intでない、値が有限の数でない、空の行）があるとき, the 診断証拠 shall 証拠も計数も変える前に拒否する。
3. The Recalibration shall 損失の列の計算と検査を、最初の状態更新より前に済ませる。
4. The Recalibration shall 並行した呼出しでの結果を保証しない。

### Requirement 6: 新実装だけで動くことと依存の向き

**Objective:** As a 研究者, I want 再較正つきのラウンドが、旧実装なしで動いてほしい, so that 後続のspecへ進める

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、ラウンドごとに、配布の後で全clientの再較正を行い、再較正が1回以上、空でない列で行われたことを確かめられる。
2. The 再較正 shall 新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
