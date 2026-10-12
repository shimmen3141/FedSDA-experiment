# Requirements Document

## Introduction
FedSDAのclientが採用した新しいモデルが、サーバへ登録されないまま残る不具合（[LEGACY-016](../../../docs/research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)）を、新実装で直す。2026-10-13にユーザーが採用を決めた（最優先）。同日の計測（提案構成の8 run）では、採用48件のうち、送信までの待ちの間に現行モデルが戻って欠番になったものが9件、次の採用で送信保留が上書きされたものが4件あった。

旧実装（固定748c3aa）との照合を、旧実装を外すまで続けられるように、旧と同じ扱いも設定で選べる形にする（2026-10-13ユーザー了承）。

## Boundary Context
- 対象: 新FedSDAのclientの送信保留、サーバの登録と正式IDの確認、その設定、基準版のgolden、修正の影響の計測。
- 対象外: 固定旧748c3aaと旧golden、FedDrift・FedAvgの動作、ほかのLEGACY、候補の採否・検出・学習の方式、旧実装の削除。
- 用語: 「送信保留」は、clientが採用して一時ID（負）で保有し、サーバの登録を待っているモデル。「待ち」は、採用から登録できるようになるまでのラウンド境界の回数。

## Requirements
### Requirement 1: 採用したモデルを、それ自身のIDで登録する
**Objective:** As a 研究者, I want 採用したモデルが必ず登録されること, so that 学習した概念のモデルが、ほかのclientへ共有される。
#### Acceptance Criteria
1. When 待ちの済んだ送信保留があり、clientの現行モデルが別の（非負または別の一時IDの）モデルである, the FedSDA system shall 送信保留のモデルの一時IDを正式IDへ付け替え、現行モデルを変えない。
2. When 待ちの済んだ送信保留のモデルが現行モデルである, the FedSDA system shall 一時IDを正式IDへ付け替え、現行モデルを正式IDへ変える（旧と同じ結果）。
3. The FedSDA system shall 付替えを、保有モデルの学習状態・損失統計・学習標本・評価標本・学習量と割当概念の計数の全部へ行う。
4. If 待ちの済んだ送信保留が無い、送信保留のモデルが保有されていない、または一時ID（負）でない, the FedSDA system shall 確認のどの状態も変える前に拒否する（正式IDが、あるownerで使用済みの場合の拒否は、各ownerの付替えの操作が行う既存の契約のままで、この項目の対象外）。
5. Where 既定の扱いを選んだ, the FedSDA system shall 採番した正式IDのそれぞれについて、登録の処理が戻った時点で、そのIDのモデルが登録したclientに保有されている状態にする（欠番を作らない。同じ同期の中の統合で、ほかのモデルへまとめられたものは、その後は、統合先のIDで保有される）。
6. When 候補検証が進行中のclientで、保有モデルの一時IDが正式IDへ付け替わる, the FedSDA system shall その検証の参照モデルの対応を付け替えず、付け替わったモデルを、その検証の再利用の対象から外す（統合でIDが付け替わるとき、および、旧と同じ扱いで現行モデルが付け替わるときと、同じ扱い）。

### Requirement 2: 待ちの間に次の採用があっても、前のモデルを失わない
**Objective:** As a 研究者, I want 連続した採用の全部が登録されること, so that 採用の間隔に依らず、モデルが共有される。
#### Acceptance Criteria
1. When 送信保留がある間に、同じclientが別のモデルを採用する, the FedSDA system shall 両方を送信保留として保持し、それぞれの待ちを、採用の時点から数える。
2. When 1回の同期で、1つのclientに待ちの済んだ送信保留が複数ある, the FedSDA system shall 採用の順に、それぞれへ正式IDを採番して登録する。
3. The FedSDA system shall clientの順、その中では採用の順に、正式IDを採番する。

### Requirement 3: 旧と同じ扱いを、設定で選べる
**Objective:** As a 開発者, I want 旧実装と同じ登録の扱いを選べること, so that 旧実装を外すまで、新旧の照合と、移植し忘れの追加の検証を続けられる。
#### Acceptance Criteria
1. Where 旧と同じ扱いを選んだ, the FedSDA system shall 送信保留を1つだけ保持し（次の採用で置き換える）、正式IDの確認で現行モデルを付け替える（現行が非負なら送信保留を外すだけ）、旧と同じ結果にする。
2. The FedSDA system shall 登録の扱いを、clientの設定の1つの値として持ち、保存するrun設定へ含め、宣言した値以外を拒否する。実験のコマンドの引数には出さない。
3. The FedSDA system shall 最終構成と、その派生の構成の既定を、要求1・2の扱いにする。
4. The FedDrift system and the FedAvg system shall これまでと同じ動作を保つ（FedDriftは、送信保留を1つだけ保持し、現行モデルを付け替える扱いを、明示して使う）。

### Requirement 4: 基準と照合を、修正の前後で分けて保つ
**Objective:** As a 開発者, I want 修正前の基準が通り続けること, so that 意図した差と、意図しない差を、区別できる。
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、旧と同じ扱いの設定で、変わらず成功させる。
2. The validation shall 基準版1のgoldenを、旧と同じ扱いの設定で、値を変えずに照合し続ける。
3. The validation shall 既定の扱いでの値を、基準版2のgoldenとして、Windowsの基準環境とWSLの両方で足す。基準版2と基準版1の差が、登録の扱いで説明できるケース（旧と同じ扱いの実行で、登録されなかった採用があるケース）に限られることと、そのようなケースが少なくとも1つあり、そこで実際に差が出ることを確かめる。
4. The validation shall 既定の扱いの全体runで、同期のたびに、登録の処理が戻った時点（待ちを進める前）で、どのclientにも待ちの済んだ送信保留が残らないことと、runの終わりに、どのclientにも、登録されなかった採用（送信保留でない、一時IDの保有モデル）が無いことを確かめる。runの終わりに送信保留のまま残るモデル（最後の区間の採用など。以後の同期が無いので登録されない。旧と同じで、終端の手順は変えない）は、登録されなかった採用とは別に数える。同じ条件の、旧と同じ扱いの実行では、登録されなかった採用があること（条件が、その経路を通ること）も確かめる。
5. The validation shall FedDriftの全体runで、正式IDの確認のたびに、送信保留のモデルが現行モデルであることを確かめる（FedDriftで、2つの扱いの結果が分かれないことの根拠）。

### Requirement 5: 修正の影響を、測って報告する
**Objective:** As a 研究者, I want 修正で精度や通信量がどう変わるかを知ること, so that 論文の実験をやり直す範囲を判断できる。
#### Acceptance Criteria
1. The validation shall 2026-10-13の計測と同じ8条件で、旧と同じ扱いと既定の扱いを実行し、精度・定常精度・通信量・グローバルモデル数・登録されなかった採用の数・runの終わりに送信保留のまま残った数を比べて、記録する。
2. If 既定の扱いで精度などが下がる条件がある, the report shall その条件と差を、ユーザーへ伝える。
