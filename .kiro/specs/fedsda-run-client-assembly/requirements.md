# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 標本1件の処理（予測を含む）は`process_observed_sample`の1回の呼出しで行えるが、それが受け取る18個のownerと16個の設定・値を作る処理がない（testと共用scriptが、その場で1つずつ作っている）。単一runの実行の枠（single-run-execution）は、clientへ5つの操作（標本の処理、ラウンド境界での保留中の学習、登録できるモデルの有無、同期後の送信待ちの進行、終端での未完了の候補検証の回収）を求める契約`RunClientOperations`を持つが、それを満たす実体がない。

変えたいこと: 最終構成のFedSDAのclientを1つ、初期モデル・初期の損失統計・設定から組み立て、実行の枠が求める5つの操作を提供する。組立てのときに、設定・閾値と、渡されたものどうしの整合を確かめる。生成直後から、標本列・ラウンド境界・終端を通して、実旧の最終構成のclient（実`__init__`で作ったもの）と状態が一致することを確かめる。

調査で確かめたこと（固定旧748c3aa）:

- 旧は、`experiment.py`の`_setup_server_and_clients`が、事前学習した1つの初期モデルと統計を、全clientへ渡す。clientは、`__init__`で初期モデルを`deepcopy`する（モデルが持つoptimizerと、その蓄積した状態も写る）。現行モデルのIDは0、一時IDは`-100 − client_id`から始まる。
- 旧の標本処理の外でclientが呼ばれるのは、ラウンド境界の`flush_pending_updates`、`has_pending_model`、同期後の`promote_pending_to_ready`、終端の`finalize_incomplete_forward_validation`（`experiment.py` 60〜93行、2283〜2287行）。新には、それぞれに当たる部品がある。
- 新の観測標本`ObservedSample`は、2特徴（float32へ丸めた値のfloat）と0/1のラベルを持ち、真の概念を含まない。実行の枠は、真の概念の列を、clientの準備の後に作る。
- 旧の値のうち、新の機能別の設定型に置き場所がまだないものがある: 許容する平均損失の増加量（旧`distance_threshold`）、候補の平均損失の最小改善量、送信までの待ちラウンド数、変化区間の最小件数、学習のbatchの件数、評価標本の保持数、検出器の候補数の上限と賭け率、検出器の表示名。
- 実旧の最終構成のclientは、実`__init__`で作って、サーバなしで、標本処理・境界・終端を続けて実行できる（下書きで確認）。

## Introduction

研究者が、最終構成のFedSDAのclientを、初期モデル・初期の損失統計・設定から1回の呼出しで組み立て、単一runの実行の枠が求める操作で動かせるようにする。組み立てたclientが、生成直後から終端まで、実旧のclientと同じ状態で進むことを確かめられるようにする。

## Boundary Context

- **In scope**: clientが受け取る設定と値の束と、その検査。ownerの生成と、渡されたものどうしの整合の検査。実行の枠が求める5つの操作（標本の処理、ラウンド境界での保留中の学習、登録できるモデルの有無、同期後の送信待ちの進行、終端での未完了の候補検証の回収）。観測標本から、標本1件の処理が受け取る形への変換。
- **Out of scope**: 初期モデルの事前学習と初期統計の計算。全clientとサーバを準備する処理（実行の枠の`prepare_run`）。サーバ同期（モデルの回収・配布、正式IDの確認、集約後の予測重みの再較正、サーバ評価）。真の概念IDを実行の枠からclientへ渡す経路（本specのclientは、任意の引数として受け取れるようにするだけで、実行の枠は変更しない）。完全なrun設定（保存表現、preset、束の値の機能別の設定型への配置）。計算量と所要時間の記録、モデル別の除外寄与の診断、標本ごとの結果種別の列、記録の保存。最終構成以外の方式。
- **Adjacent expectations**: 標本1件の処理、保留中の学習要求の学習、送信保留、未完了の候補検証の回収は、移植済みの部品をそのまま使い、中の処理を変えない。初期モデルは、分類器と、その概念固有部・共有部のoptimizerの状態の組で渡される（事前学習のspecが作る）。乱数生成器は、runが作ったものを借りる。

## Requirements

### Requirement 1: clientの組立て

**Objective:** As a 研究者, I want clientを、初期モデル・初期の損失統計・設定から1回で組み立てたい, so that 標本1件の処理が受け取るownerと値を、呼出し側が1つずつ作らなくてよい

#### Acceptance Criteria

1. When clientの組立てを求められたとき, the Client Assembly shall 標本1件の処理が受け取る全ownerを新しく作り、初期モデルだけを保有し、それを現在の学習帰属とするclientを返す。
2. When clientを組み立てたとき, the Client Assembly shall 渡された初期モデル（分類器と2つのoptimizerの状態）の、値と蓄積した状態が同じ独立した写しをclientへ持たせ、渡された初期モデルを変えない。同じ初期モデルから組み立てた複数のclientは、互いに影響しない。
3. When clientを組み立てたとき, the Client Assembly shall 生成直後の状態を、同じ初期モデル・統計・設定から実`__init__`で作った実旧の最終構成のclientと一致させる: 保有モデルとoptimizerの状態、損失統計、損失の監視の基準、一時IDの開始値、空の保留・学習データ・記録、送信保留なし、学習要求なし。
4. The Client Assembly shall 組立ての間、乱数を消費しない。

### Requirement 2: 標本の処理

**Objective:** As a 研究者, I want 観測標本と位置を渡すだけで、標本1件の処理が行われてほしい, so that 実行の枠が、clientの中のownerを知らずに標本を進められる

#### Acceptance Criteria

1. When 観測標本と標本の位置を渡されたとき, the Run Client shall 観測標本を、特徴が1行の2次元・ラベルが1行1列のfloat32の値へ変換し、標本1件の処理（予測を含む）を1回行って、その結果を返す。
2. Where 真の概念IDが渡されたとき, the Run Client shall それを診断用の概念IDとして標本1件の処理へ渡す。渡されなければ、概念IDなしで処理する。
3. If 観測標本が決まった型でないとき、または位置・真の概念IDが整数でないとき, the Run Client shall どの状態も変える前に拒否する。

### Requirement 3: ラウンド境界と終端の操作

**Objective:** As a 研究者, I want ラウンド境界と終端の処理を、clientの操作として呼びたい, so that 実行の枠の順序どおりに、旧と同じ処理が行われる

#### Acceptance Criteria

1. When ラウンド境界での保留中の学習を求められたとき, the Run Client shall 保留中の学習要求があれば学習し、なければ何もしない。
2. When 登録できるモデルの有無を尋ねられたとき, the Run Client shall 送信保留のモデルがあり、送信までの待ちが済んでいるときだけ、真を返す。状態は変えない。
3. When 同期後の送信待ちの進行を求められたとき, the Run Client shall 送信保留のモデルの待ちを1ラウンド進める。送信保留がないとき、または既に送信できるときは、何もしない。
4. When 終端での未完了の候補検証の回収を求められたとき, the Run Client shall 保持中の候補検証があれば、それを回収して適応記録を足し、候補検証へ渡した標本の概念IDの保持を外す。保持していなければ、何もしない。
5. If ラウンドの番号が非負の整数でないとき, the Run Client shall どの状態も変える前に拒否する。

### Requirement 4: 実旧との一致

**Objective:** As a 研究者, I want 組み立てたclientが、最初の標本から終端まで、実旧のclientと同じ状態で進んでほしい, so that 新clientを、全体runの接続へ進める

#### Acceptance Criteria

1. When 同じ初期モデル・統計・設定・標本列・乱数を与えたとき, the Run Client shall 実`__init__`で作った実旧の最終構成のclient（サーバなし）と、標本ごと・ラウンド境界ごと・終端の後に、保有モデルとoptimizerの状態、学習データ、計数、損失統計、保留、監視、警報の記録、適応記録、予測の記録と重み、診断証拠、送信保留、学習要求、乱数の状態が一致する。
2. The Run Client shall 上の対照で、警報の各結果（不足、現行の維持、他モデルの再利用、候補検証の開始、候補検証中の警報）、候補検証の確定（採用、棄却）、送信待ちの進行、終端での未完了の候補検証の回収を通る。

### Requirement 5: 設定と整合の検査

**Objective:** As a 研究者, I want 不正な設定や、合わないものどうしが、組立てのときに拒否されてほしい, so that 標本の処理の途中で初めて拒否されて、途中まで更新された状態が残ることがない

#### Acceptance Criteria

1. If 設定の束の値のいずれかが、型または範囲（増加量と改善量は有限の非負の数、待ちラウンド数・最小件数・batchの件数・候補数の上限・評価標本の保持数の上限は1以上の整数、評価標本を1回に足す件数は非負の整数、表示名は空白でない文字列、賭け率は0より大きく1未満の数の空でない列）に合わないとき, the Client Assembly shall clientを作らずに拒否する。
2. If 機能別の設定のいずれかが決まった型でないとき, the Client Assembly shall clientを作らずに拒否する。
3. If 最終構成で同じ値でなければならない組（Fixed-Shareの時間尺度と保留の容量、候補の平均損失の最小改善量と候補の学習の早期終了の最小改善量）が違うとき, the Client Assembly shall clientを作らずに拒否する。
4. If 初期モデルの2つのoptimizerの状態が、初期の分類器の概念固有部・共有部のパラメータと対応しないとき、分類器の入力の特徴数が観測標本の特徴数（2）と違うとき、または初期の損失統計・client ID・初期モデルのID・乱数生成器が決まった型と範囲でないとき, the Client Assembly shall clientを作らずに拒否する。
5. The Client Assembly shall 拒否のとき、渡された初期モデル・統計・乱数生成器を変えない。
6. The Run Client shall 並行した呼出しでの結果を保証しない。

### Requirement 6: 実行の枠への適合と、新実装だけで動くこと

**Objective:** As a 研究者, I want 組み立てたclientが、既存の実行の枠でそのまま動いてほしい, so that 残りはサーバ側と準備の処理だけになる

#### Acceptance Criteria

1. When 組み立てたclientを参加者として渡したとき, the 実行の枠 shall 参加者の検査を通し、区間の進行（標本の処理、ラウンド境界、終端）を、clientの操作を呼んで最後まで実行できる（サーバの操作は、何もしない代役でよい）。
2. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、clientを組み立て、実行の枠で標本列を最後まで進められる。
3. The Client Assembly shall 新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
