# 要求: 分類器のparameter snapshot

## Project Description
研究実装の開発者が、準備済み分類器の現在値を候補初期化や送信準備へ渡すため、独立した全parameter snapshotを生成できるようにする。既存の選択・平均部品はそのsnapshotを受け取り、送信保留状態は後続で管理する。

## Boundary Context
最終Residual Adapter構成の通常モデルを対象とする。共有部・残差部・分類層の全parameterを含み、optimizer・勾配・実行状態をsnapshotへ含めない。外部hookによる副作用やモデル構造の改ざん・並行変更は対象外。

## Requirements
### Requirement 1: 現在値の独立した記録
1.1 When 適合する分類器のsnapshotが要求される, the snapshot生成部 shall 共有特徴抽出部・残差部・分類層の全parameter名と現在値をモデルの列挙順で返し、形状と値を変更しない。
1.2 When snapshotを返す, the snapshot生成部 shall 勾配記録のない独立結果を返し、返却後のモデル更新とsnapshot更新が互いの値に波及しない。
1.3 When 同じモデルのsnapshotを繰り返し要求する, the snapshot生成部 shall 各呼出時点の値を、それ以前の結果から独立して返す。

### Requirement 2: 入力と状態の保護
2.1 If 分類器の型またはparameterの実行環境・有限性が契約に適合しない, the snapshot生成部 shall 項目名と理由を含む例外で拒否し、モデル値と既存gradientを変更しない。
2.2 The snapshot生成部 shall forward・学習・optimizer resetを行わず、通常モデルのparameterと既存gradient・training flags・共有参照・乱数状態・呼出元の勾配記録設定を保持する。

### Requirement 3: 移植と責務
3.1 When 同一条件の旧モデル値からsnapshotを作成する, the 移植検証 shall 新旧parameter名を明示対応させ、全parameterの形状・値・列挙順とモデルからの独立性を一致させる。
3.2 The snapshot生成部 shall parameter値の取得と独立コピーだけを行い、候補初期化元の選択・平均・モデルへの適用・登録・統計保存・送信保留状態の管理を行わない。
3.3 The 本移植 shall 固定旧実装と既存goldenの値・許容差を変更せず、構造整理にアルゴリズム変更を含めない。
