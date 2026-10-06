# 調査と判断

## Summary
2026-10-06。追加依存なしの既存PyTorch拡張。旧BaseClient._register_trained_new_model（clients/base.py:286）はprepare→models登録→no_grad per_sample_error→batch統計→pending snapshotの順。今回の切出しはper_sample_errorだけである。

## Research Log
- models.py:66のSimpleMLP.per_sample_errorは二値abs(forward−label).view(-1)、多クラスsoftmax(dim=1)→正解gather→1−確率。SharedHeadMLPは継承し、Residual Adapter出力も同じ規約。
- 新ResidualAdapterClassifierは二値sigmoid出力、多クラスlogit、CPU float32明示。通常層はLinear/ReLUでdropoutや可変running bufferを持たない。
- class_probability_calculationsはモデルを呼ばず予測済み確率から平均を作る部品。ID付き辞書へ架空IDを入れて使うより、一モデルの標本別結果を返す独立関数が境界に適合する。既存部品を改変しない。
- batch-loss-statistics-initializationは外部損失shape[N]とラベルshape[N,1]のCPU float32を受理。今回の出力とそのまま接続できる。
- held-model-training-state-registryは保有参照の管理だけで、今回の損失評価へ依存させない。

## Design Decisions
Generalization: 採用候補だけに限定せず準備済み一分類器のbatch評価。異なるモデル方式の追加は将来の明示変更。
Build vs adopt: 既存PyTorch公開演算で旧演算順を移植。損失生成を学習目的関数BCELoss/CrossEntropyLossと混同しない。新ライブラリ/汎用Protocol/評価サービスを増やさない。
Simplification: 一関数・二検証helper、状態record/設定/IDは不要。通常モデルのみを契約とし、hookによる副作用・外部による構造の改ざん・並行変更を保証範囲に含めない。

## 使用した手順
fable-method、kiro-spec-init、kiro-spec-requirements（EARS/要求gate）、kiro-spec-design（light discovery/synthesis/design gate）、kiro-spec-tasks、kiro-reviewを参照。新規ライブラリや外部APIの調査は不要。
