# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: client側は、事前学習→clientの組立て→実行の枠の5操作まで動くが、サーバ側は何もない（testと共用scriptは、何もしない代役を使っている）。clientが採用した新規モデルは、送信保留のまま登録されず、clientどうしのモデルは集約されない。

変えたいこと: 旧の最終構成のサーバ（`SharedBackboneFedSDANoCachedServer`）の1ラウンドの前半——新規モデルの登録（正式IDの採番、来歴の記録、clientへの正式IDの確認）と、集約（共有部1個と概念固有部ごとのFedAvg、損失統計の平均、通信量の記録）——を、旧と同じ順序と数値で行う。サーバが持つ状態（グローバルモデル、統計、次の正式ID、来歴、通信量）のownerを作る。

調査で確かめたこと（固定旧748c3aa）:

- 旧のラウンドは、`run_round`（`servers/fedsda.py` 49〜70行、`servers/shared_backbone.py` 25〜30行）が、新規モデルの登録→集約→（新規モデルがあるラウンドだけ）クロス評価・クラスタリング・統合→配布→集約後の予測重みの再較正、の順に行う。本specは、最初の2つを扱う。
- 登録（`_register_new_models`、`servers/fedsda.py` 72〜87行）: client順に、送信できるモデルを持つclientへ、正式IDを採番し、来歴（モデルID、ラウンド、client）を記録し、clientへ正式IDを確認させる。パラメータは送らない（集約のときに1回だけ送る）。
- 集約（`update_global_models`、`servers/shared_backbone.py` 54〜130行）: 対象は、いずれかのclientが保有する非負のIDのモデル。clientごとに、学習データを1件以上持つモデルだけが参加する。共有部は、clientごとに1回、そのclientの参加モデルの学習データの総件数で重み付けて平均する。概念固有部と損失統計の平均は、モデルIDごとに、学習データの件数（統計は統計の件数）で重み付けて平均する。通信量は、clientごとに共有部1回と、参加モデルごとの概念固有部1回を数える。
- 旧のサーバは、グローバルモデルを、モデルIDごとの完全なパラメータ（共有部を含む）として持つ。集約の対象にならなかったモデルは、古い共有部の値のまま残る。
- 旧の挙動（LEGACY-016）: clientが候補を採用した後、送信できるようになる前に現行モデルが非負のIDのモデルへ戻ると、サーバは正式IDを採番して来歴を記録するが、clientは送信保留を外すだけで、採用したモデルは一時IDのまま残る（登録も集約もされない）。移植済みの正式IDの確認（held-model-registration-confirmation）は、この挙動を維持している。
- 実旧のサーバは、登録と集約だけを、配布なしで、ラウンドごとに続けて実行できる（下書きで確認）。

## Introduction

研究者が、サーバの1ラウンドの前半（新規モデルの登録と、clientのモデルの集約）を、旧と同じ結果になる呼出しで行えるようにする。

## Boundary Context

- **In scope**: サーバが持つ状態のowner（グローバルモデルのパラメータと損失統計、次の正式ID、登録の来歴）。通信量の記録のowner。新規モデルの登録。clientのモデルの集約。
- **Out of scope**: クロス評価・クラスタリング・統合、モデルの配布とclient側の反映、集約後の予測重みの再較正（それぞれ後続のspec）。実行の枠のサーバの操作（`RunServerOperations`）を満たす実体と、全clientとサーバの準備（全体runを接続するspec）。クラスタリングの診断の記録（来歴のうち、登録だけを扱う）。最終的なモデルの容量の集計。最終構成以外のサーバ（共有部を持たない方式、Cached）。
- **Adjacent expectations**: clientは、組立て済みの`FedsdaRunClient`で、ownerを読める。正式IDの確認、パラメータの写し、損失平均の集約は、移植済みの部品をそのまま使い、中の処理を変えない。初期モデルのパラメータと統計は、事前学習の結果から渡される。

## Requirements

### Requirement 1: サーバが持つ状態

**Objective:** As a 研究者, I want サーバの状態を、1つのownerから読みたい, so that 登録・集約・後続の配布と統合が、同じ状態を扱える

#### Acceptance Criteria

1. When グローバルモデルのownerを作るとき, the Global Model Repository shall 初期モデルのIDで、そのパラメータの写しと損失統計を持ち、初期モデルの来歴を「ラウンドなし・clientなし」として記録し、次の正式IDを初期モデルのIDの次にする。
2. When 正式IDの採番を求められたとき, the Global Model Repository shall 次の正式IDを返し、次の正式IDを1つ進める。
3. The Global Model Repository shall モデルIDごとのパラメータと損失統計を、設定した順に保持し、渡された値・返す値を、内部と結合しない写しにする。
4. The Global Model Repository shall 登録の来歴（モデルID、ラウンド、client）を記録し、モデルIDの昇順で返す。同じモデルIDへの記録は、後の記録で置き換える。
5. If モデルID・ラウンド・clientのID、パラメータ、損失統計が、決まった型と範囲でないとき, the Global Model Repository shall 何も変えずに拒否する。

### Requirement 2: 通信量の記録

**Objective:** As a 研究者, I want 通信量を、旧と同じ項目で数えたい, so that 通信量の指標を旧と比べられる

#### Acceptance Criteria

1. The Communication Volume Record Store shall 上りと下りのそれぞれについて、論理的なモデル転送数、軽量メッセージ数、パラメータの値の数、バイト数を、0から数える。
2. When パラメータの転送を記録するとき, the Communication Volume Record Store shall そのパラメータの値の数とバイト数（値の数×1値のバイト数）に転送回数を掛けて、指定の向きへ足す。
3. If 向き・件数・パラメータが、決まった型と範囲でないとき, the Communication Volume Record Store shall 何も変えずに拒否する。

### Requirement 3: 新規モデルの登録

**Objective:** As a 研究者, I want 送信できる新規モデルへ、旧と同じ順で正式IDが付いてほしい, so that モデルIDの付き方が旧と同じになる

#### Acceptance Criteria

1. When 新規モデルの登録を求められたとき, the Server Model Registration shall clientを渡された順に見て、送信できるモデルを持つclientごとに、正式IDを採番し、来歴（そのモデルID、ラウンド、clientのID）を記録し、clientへ正式IDを確認させる。
2. When clientの現在の学習帰属が一時IDのモデルであるとき, the Server Model Registration shall clientの中のそのモデル（保有モデル、損失統計、学習データ、評価標本、計数、現在の学習帰属）を正式IDへ付け替え、送信保留を外す。
3. When clientの現在の学習帰属が非負のIDのモデルであるとき, the Server Model Registration shall 採番と来歴の記録は行い、clientの中では送信保留を外すだけにする（旧の挙動の維持）。
4. The Server Model Registration shall 登録では、通信量を足さず、グローバルモデルのパラメータと統計を変えない。
5. The Server Model Registration shall 登録の結果（clientのID、採番した正式ID、clientの学習帰属が変わったか）を、登録した順に返す。

### Requirement 4: clientのモデルの集約

**Objective:** As a 研究者, I want clientのモデルが、旧と同じ重みと演算の順で集約されてほしい, so that グローバルモデルと統計が旧と同じ値になる

#### Acceptance Criteria

1. When 集約を求められたとき, the Server Model Aggregation shall いずれかのclientが保有する非負のIDのモデルを、IDの昇順で、集約の対象にする。
2. When 集約するとき, the Server Model Aggregation shall clientごとに、対象のモデルのうち、そのclientが学習データを1件以上持つものだけを参加させる。参加するモデルのないclientは、集約に寄与しない。
3. When 集約するとき, the Server Model Aggregation shall 共有部を、clientごとに1回、そのclientの参加モデルの学習データの総件数で重み付けて平均し、概念固有部を、モデルIDごとに、各clientのそのモデルの学習データの件数で重み付けて平均する。
4. When 集約するとき, the Server Model Aggregation shall 対象のモデルごとに、グローバルモデルのパラメータを、平均した共有部と、そのモデルの概念固有部（参加したclientがなければ、既存のグローバルモデルの概念固有部）の組にする。参加したclientがなく、既存のグローバルモデルもないモデルは、作らない。どのclientも参加しなかったときの共有部は、既存の最初のグローバルモデルの共有部にする。
5. When 集約するとき, the Server Model Aggregation shall 対象のモデルごとに、参加したclientの損失統計の平均を、統計の件数で重み付けて平均し、件数の合計が1以上なら、グローバルの損失統計を、その件数と平均（偏差平方和は0、クラス別の統計なし）に置き換える。
6. When 集約するとき, the Server Model Aggregation shall 上りの通信量へ、clientごとに、参加モデルの数だけの論理的なモデル転送、共有部のパラメータ1回、参加モデルごとの概念固有部のパラメータ1回を足す。
7. The Server Model Aggregation shall 対象のモデルIDごとの、参加した学習データの総件数（参加がなければ0）を返す。clientの状態は変えない。

### Requirement 5: 実旧との一致

**Objective:** As a 研究者, I want 登録と集約が、複数のclientと複数のラウンドで、旧と同じ結果になってほしい, so that 後続の配布と統合を、同じ状態からつなげる

#### Acceptance Criteria

1. When 同じ初期モデル・統計・設定・標本列・乱数を与えたとき, the 登録と集約 shall 実旧のサーバと、実`__init__`で作った実旧のclient複数に対して、ラウンドごとに、グローバルモデルのIDの順・全パラメータ・損失統計、次の正式ID、登録の来歴、通信量の全項目、集約の件数、各clientの全状態、乱数の状態が一致する。
2. The 対照 shall 一時IDのモデルの登録、非負のIDへ戻った後の登録（LEGACY-016）、複数のclientが同じモデルを持つ集約、学習データを持たないモデルを含む集約を通る。

### Requirement 6: 拒否と途中の失敗

**Objective:** As a 研究者, I want 不正な入力が、状態を変える前に拒否されてほしい, so that サーバとclientの状態が、途中まで更新されたまま残らない

#### Acceptance Criteria

1. If clientの列が決まった型のtupleでないとき、clientのIDが重複するとき、ownerが決まった型でないとき、またはラウンドが非負の整数でないとき, the 登録と集約 shall どの状態も変える前に拒否する。
2. The Server Model Aggregation shall グローバルモデル・統計・通信量の更新を、全部の計算が済んでから行う（計算の途中で失敗したら、何も変えない）。
3. If 登録の途中で、あるclientへの確認が失敗したとき, the Server Model Registration shall 後のclientへ進まない。済んだclientの登録と、失敗したclientへの採番・来歴は残る（巻戻しなし）。
4. The 登録と集約 shall 並行した呼出しでの結果を保証しない。

### Requirement 7: 新実装だけで動くことと依存の向き

**Objective:** As a 研究者, I want 登録と集約が、旧実装なしで動いてほしい, so that 後続のspecへ進める

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、複数のclientの標本処理の後で、登録と集約をラウンドごとに実行できる。
2. The 登録と集約 shall 新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
