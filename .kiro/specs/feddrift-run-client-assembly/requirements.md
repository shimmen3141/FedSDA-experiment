# Requirements Document

## Project Description
実験コードの保守担当はFedDriftを新構成へ移植している。検出バッチ・観測列・round学習の部品は移植済みだが、初期モデルからclientを組み立てて一連の操作を実行する入口がない。初期モデル0・optimizer状態・損失統計を独立に写し、観測ブロック、round学習、登録可否照会、終端flushを同じownerへ接続する。実旧clientとの状態・乱数の一致を示し、サーバ同期と全体runの制御は後続へ引き継ぐ。

## Introduction
実験担当が、独立モデルを使うFedDrift clientを初期状態から動かし、旧clientの生成・観測・学習・終端の流れと比較できるようにする。

## Boundary Context
- 含む: 一clientの組立、機能別の設定、観測ブロック・一round学習・登録可否・残余検出の接続。
- 含まない: 初期事前学習の実行、全client factory、サーバ通信/配布/クラスタリング、R回の全体進行、演算/時間計測、全体runのgolden移植。
- 隣接への期待: 初期モデル0と学習済みoptimizer・統計を呼出し側が渡す。サーバを含む呼出し側は旧の観測→準備→R回学習/同期を保つ。既存FedSDAループへ同じ操作を当てはめない。

## Requirements

### Requirement 1: 初期状態の独立した組立
**Objective:** As a 実験担当, I want 初期状態から独立したclientを作る, so that runとclient間の更新が干渉しない。

#### Acceptance Criteria
1. When 初期モデル0・optimizer状態・統計を渡す, the client組立 shall parameter値・optimizer状態・train/eval状態を保ち初期勾配を旧同様Noneにした独立した写しと、空の学習/評価/計数/送信/観測状態を作り、現在ID0・一時ID開始値-100-client ID・前回最小損失なしとする。
2. If client ID、設定、モデル構造/parameter、optimizer参照、初期統計が不正, then the client組立 shall 元の入力状態・乱数を変更せず生成を拒否する。
3. When 同じ初期状態から複数clientを作る, the client組立 shall 初期入力と他clientの可変状態から独立し、組立でPython・NumPy・torch乱数を消費しない。

### Requirement 2: 旧の条件に対応する設定
**Objective:** As a 実験担当, I want 必要な条件を明示的に渡す, so that 大域設定を切り替えずclientを動かせる。

#### Acceptance Criteria
1. When 条件を指定する, the client設定 shall 非負有限の損失増分閾値、正の検出/学習batch件数・評価保持上限、非負の更新倍率・評価追加件数、有効な新規optimizer条件を受理し、bool・型/値域違反を拒否する。

### Requirement 3: clientの観測とround進行
**Objective:** As a 実験担当, I want 組み立てたclientで一連の処理を行う, so that 既存の部品が同じ状態に作用する。

#### Acceptance Criteria
1. When 観測ブロックを渡す, the client shall 全ブロックを状態更新前に検査し、予測→蓄積→満杯検出の順に進め、検出が一度でも完了したかを返し、通常観測だけではparameterを学習しない。
2. When 一round学習を依頼する, the client shall 検出batch件数を倍率として旧local_trainと同じモデル順・抽出・損失・更新・計数・乱数消費で学習し、既存のownerを保持する。
3. When 終端flushを依頼する, the client shall 残余検出batchを一度だけ処理し、過去の予測を変更せず、学習を追加実行しない。
4. When 真の概念IDだけを変える, the client shall 予測の診断記録だけを変え、予測/検出/割当/学習と乱数を変えない。
5. When 登録可否を照会する, the client shall 同じ送信状態のready値に対応するboolを返し、モデル・乱数・送信状態を変更しない。

### Requirement 4: 新構成だけの接続
**Objective:** As a 保守担当, I want 新構成だけでclientを動かす, so that 旧実装を将来外せる。

#### Acceptance Criteria
1. The client実装 shall 旧実装・大域config・掃引・保存・testへ依存せず、fresh processで生成→観測→round学習→終端の接続を実行できる。
