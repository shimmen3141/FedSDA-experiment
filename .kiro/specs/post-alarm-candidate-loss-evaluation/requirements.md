# 要件: 警報後の候補損失評価

## Introduction

研究者が最終FedSDAの警報後候補採否を、旧基準と同じ比較対象・損失値・判定結果で確認できるようにする。

## Boundary Context

外部で収集済みの候補/参照モデルの同じ観測順の有界損失列、警報時点の履歴平均、現在利用可能なモデルID、現行学習モデルID、明示判定条件を入力する。既存の最終採否方針と将来検証件数を使用する。
この部品はモデルを再学習しない。「適合する既存モデル」の選択は履歴平均との差で確認するだけであり、実際の再利用/候補登録/割当変更は呼出側が実施する。
損失収集のsession・モデルforward・shadow生成・学習・履歴平均推定・FIFO・警報後操作・新全体runは対象外。最終方針以外の旧policyは移植しない。

## 入力・結果の契約

- 候補損失は不変な列、参照損失はIDから同長の不変な列への非空対応。全値はbool以外のbuiltin int/float、有限で0～1。件数は既存設定の検証件数以上、2以上。
- IDはbool以外のbuiltin intで、負IDも許可する。利用可能IDは重複なしの不変な列。現行IDが参照・利用可能集合にない場合も、他の適合参照または全参照最小損失との比較を行う。
- 履歴平均はIDから有限[0,1]値への対応。参照の履歴がない場合は再利用適合性の判定から除外する。参照以外の履歴値も入力として保持できる。
- 再利用の履歴平均超過許容値と候補の必要改善量はbool以外の有限builtin int/float、0以上。
- 結果は比較対象ID、既存モデル適合ID（なければなし）、候補採否、理由、件数、候補/比較対象の全体と後半平均、比較対象履歴平均（なければなし）。二分区間の前半はfloor(N/2)、後半は残り。

## Requirements

### Requirement 1: 完了済み損失の明示入力

1. The Candidate Loss Evaluation System shall 最終採否方針・検証件数と損失列・履歴平均・利用可能ID・現行学習ID・二つの判定閾値を明示入力で受け、旧グローバル設定を読まない。
2. When 検証件数を満たす同長の候補/参照損失が渡される, the Candidate Loss Evaluation System shall 観測順と参照の入力順を保持して評価する。
3. If 型・有限性・値域・件数・IDの契約に反する, the Candidate Loss Evaluation System shall 理由付きで拒否し、入力の列・対応表・設定を変更しない。

### Requirement 2: 既存モデルの再利用適合性

1. When 利用可能な参照に履歴平均がある, the Candidate Loss Evaluation System shall 将来平均から履歴平均を引いた値が許容値以下の場合に適合と判定し、閾値の等号を含める。
2. When 現行学習モデルが適合する, the Candidate Loss Evaluation System shall 他参照の平均損失が低くても現行モデルを選ぶ。
3. When 現行学習モデルが適合しない, the Candidate Loss Evaluation System shall 適合参照の最小平均損失を選び、同率では小さいモデルIDを選ぶ。適合参照がない場合はなしを返す。

### Requirement 3: 候補の二分区間採否

1. When 適合する既存モデルがある, the Candidate Loss Evaluation System shall そのモデルを比較対象として候補を棄却し、現行モデルか代替モデルかを理由で区別する。
2. When 適合する既存モデルがない, the Candidate Loss Evaluation System shall 全参照の損失合計が最小のモデルを比較対象とし、同率では入力順を維持する。利用可能集合から消えた参照もこの比較に残す。
3. When 候補を比較対象と評価する, the Candidate Loss Evaluation System shall 旧基準と同じfloat32区間平均をPython floatへ取り出し、前半と後半の両方で候補平均が参照平均から必要改善量を引いた値より厳密に小さい場合だけ採用する。
4. When 判定理由を返す, the Candidate Loss Evaluation System shall 旧基準と同じfloat32区間平均の参照−候補差を先にfloat32で求め、Python floatへ取り出して必要改善量以下か判定し、不合格区間が前半・後半・両方かを報告する。採否と理由を独立に計算し、採用なのに区間margin不合格、棄却なのに両margin合格となる丸め境界の結果も保持する。

### Requirement 4: 診断値と独立性

1. When 評価が完了する, the Candidate Loss Evaluation System shall 比較対象・適合ID・採否・理由・件数・候補/参照の全体と後半平均・参照履歴平均を変更不能な結果として返す。
2. The Candidate Loss Evaluation System shall 入力やモデル/学習状態・共有乱数を変更せず、複数呼出し間に可変状態を蓄積しない。

### Requirement 5: 旧基準と責務境界

1. The Candidate Loss Evaluation System shall 現行優先・代替適合・適合なし・同率・履歴欠落・消えた参照・奇数/偶数区間・改善閾値境界を旧最終方針へ直接照合できる。
2. The Candidate Loss Evaluation System shall public有界損失から渡す接続を独立検証でき、損失収集やモデル操作を評価へ取り込まない。
3. The Candidate Loss Evaluation System shall 旧import・互換aliasを持たず、全golden・依存境界・独立起動確認を通過し、完成を新FedSDA全体runの完成とは扱わない。
