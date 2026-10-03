# 要件: 警報後の候補・参照損失の収集

## Introduction

候補採否の数値評価へ渡す警報後の損失列を、研究者が観測順・件数・終了時機まで旧最終構成と照合できるようにする。

## Boundary Context

上位処理が候補と警報時点の参照モデルで算出した損失を入力する。収集開始、観測位置と損失列の保持、規定件数の到達、変更不能な収集結果を対象とする。
モデルforward/学習・参照モデルのsnapshot生成・履歴baseline・候補採否・正式登録/切替・payload/held_data・episode・終端回収は対象外。現在利用可能な参照ID集合は採否評価の時点に上位が供給し、収集中の参照ID集合と区別する。
旧正常client経路では警報を処理した後に収集を開始し、次標本から規定件数を収集した直後に採否評価する。この到達範囲を維持し、規定件数を超える追加は新契約で拒否する。

## 入力契約

- 既存候補方針設定の検証件数はbool以外のbuiltin int、2以上。
- 提案位置と観測位置はbool以外の非負builtin int。最初は提案位置＋1、それ以降は直前＋1。
- 開始時の参照ID列は非空・重複なしのtuple、各IDはbool以外のbuiltin int。負IDも受理し、入力順を維持する。
- 候補と各参照の損失はbool以外のbuiltin int/float、有限の[0,1]。保存時はfloat化し、平均・丸め・並替えを行わない。
- 各回の参照損失dictは開始時と完全に同じID集合を含む。dict内の順番は自由、系列の保存順は開始時ID順。
- 未完了・完了の診断copyは、提案位置、最後の検証位置（未観測ならなし）、候補損失tuple、参照IDごとの損失tuple、検証件数、規定件数、到達状態を観測可能にする。

## Requirements

### Requirement 1: 明示開始

1. The Candidate Loss Collection System shall 明示された候補設定・提案位置・参照ID列から空の収集を開始し、旧グローバル設定やモデルを参照しない。
2. When 収集を開始する, the Candidate Loss Collection System shall 開始時の参照IDと順序を固定し、呼出側の後の利用可能モデル集合変更により収集参照を変更しない。
3. If 開始条件の型・値域・ID重複/空条件に反する, the Candidate Loss Collection System shall 理由付きで拒否する。

### Requirement 2: 時系列の原子的収集

1. When 正解観測後の損失を追加する, the Candidate Loss Collection System shall 提案位置＋1から連続する観測位置だけを受理し、警報当日の標本を将来検証へ含めない。
2. When 候補と全参照の損失を追加する, the Candidate Loss Collection System shall 同じ観測回に対応する各系列へ観測順のfloat値を一件ずつ保持する。
3. If 位置・損失の型/値域・参照ID集合に反する, the Candidate Loss Collection System shall 理由付きで拒否し、どの系列・位置・到達状態も変更しない。
4. The Candidate Loss Collection System shall 入力dictの順番や呼出後の変更に影響されず、開始時の参照順と各系列の同じ件数を保持する。

### Requirement 3: 規定件数と終了境界

1. While 検証件数が規定未満である, the Candidate Loss Collection System shall 未到達として件数を観測可能にする。
2. When 規定件数へ到達する, the Candidate Loss Collection System shall 到達状態を返し、その回の観測位置を最後の検証位置として保持する。以後の追加は状態変更せず拒否する。
3. The Candidate Loss Collection System shall 到達時や実験終端に採否・モデル操作・payload回収・損失消去を自動実行しない。

### Requirement 4: 状態と副作用

1. The Candidate Loss Collection System shall 途中と到達後に独立した変更不能な診断copyを返し、参照IDと損失系列の内部可変状態を公開しない。
2. The Candidate Loss Collection System shall 実体ごとに独立状態を所有し、共有乱数や既定数値型/deviceを変更しない。

### Requirement 5: 移植の証拠

1. The Candidate Loss Collection System shall 旧sessionへの直接追加と旧clientの観測→規定件数到達の直接実行へ系列・位置・到達回を照合できる。
2. When 到達後の損失列を既存の候補採否評価へ明示入力する, the Candidate Loss Collection System shall 旧採否と同じ結果を得られる形で渡せるが、収集部品から採否評価を自動起動しない。
3. The Candidate Loss Collection System shall 旧import/互換aliasなしの依存境界・全golden・独立起動を確認し、部分完成範囲と発見した旧不具合/改善点を共通記録へ接続する。

