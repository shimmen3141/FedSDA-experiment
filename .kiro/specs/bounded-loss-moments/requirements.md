# 要件: 有界損失の逐次集計

## Introduction
監視基準・候補比較・モデル統計の前提となる件数・平均・平均からの偏差平方和を、旧実装と同じ演算順で保持・照合できるようにする。

## Boundary Context
一つの有界損失系列の不変な集計値、初期集計の受取、1観測追加、2件以上の平均/不偏標本分散の推定を対象とする。
モデル/クラスIDへの所属管理、batch集計、監視基準clipと1件閾値、候補履歴採用条件、モデル統合/再採番、学習/登録は対象外。平均と分散の推定に2件必要なことと、1件で平均自体が利用できることを区別する。
旧新規モデルseedには1件でも偏差平方和0.1が入り得る。種別を推測して補正・再計算しない。

## 入力契約
件数はbool以外の非負builtin intで、有限floatとして表現できる値。平均はbool以外のbuiltin int/floatの有限[0,1]、偏差平方和は同じ型の有限非負値。0件は平均と偏差平方和がともに0。1件以上のseedは偏差平方和を件数から追加制約せず受け取る。
観測損失はbool以外のbuiltin int/floatの有限[0,1]。不変値は入力をfloat化するがfloat32化や丸め・偏差平方和の補正を行わない。推定値は件数・平均・不偏標本分散を保持し、2件未満は推定なしで示す。

## Requirements

### Requirement 1: 不変な集計値
1. The Loss Moment System shall 明示された件数・平均・偏差平方和から独立した変更不能な集計値を構築する。
2. When 0件または1件以上の初期集計を受ける, the Loss Moment System shall 入力契約を検査し、旧1件seedの非零偏差平方和も補正せず保持する。
3. If 型・有限性・値域・空集計の整合性に反する, the Loss Moment System shall 項目名と理由を含めて拒否する。

### Requirement 2: 観測の追加
1. When 一つの有界損失を追加する, the Loss Moment System shall 件数増加→旧平均との差→平均加算→新平均との差→偏差平方和加算の順にPythonfloat演算で新しい不変集計値を返す。
2. The Loss Moment System shall 入力集計を更新せず、観測順を並替えたりbatch平均へ置換したりしない。
3. If 入力集計・損失または演算後の集計が契約に反する, the Loss Moment System shall 拒否し、入力集計を変更しない。

### Requirement 3: 平均・不偏標本分散
1. While 集計件数が2未満である, the Loss Moment System shall 推定なしを返し、真のゼロ平均/分散と区別する。
2. When 集計件数が2以上である, the Loss Moment System shall 件数・保存平均・偏差平方和/(件数−1)を変更不能な推定値として返す。
3. The Loss Moment System shall 保存された初期偏差平方和を使い、分散の再計算や非零1件seedの除去をしない。

### Requirement 4: 移植と接続の証拠
1. The Loss Moment System shall 旧Welford関数・旧全体/クラス別更新・旧平均分散取得へ各観測の全値を直接照合できる。
2. The Loss Moment System shall 別系列・返却値・共有乱数・既定dtype/deviceに副作用を与えず、上位監視と候補評価へ保存平均・推定の有無を明示入力できる。
3. The Loss Moment System shall stdlibだけの依存・旧import/互換aliasなし・全goldenと独立起動を確認し、完成境界と発見事項を正本へ記録する。

