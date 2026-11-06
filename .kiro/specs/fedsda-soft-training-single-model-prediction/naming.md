# Naming

案Aに揃えた下書き（主担当の判断、2026-11-07にユーザーが案Aを確定）。独立CLIレビューのFAILを受けた指摘反映段階であり、ユーザー確認と仕様レビュー承認までは実装しない。共通規約に従い引数・field・局所名・非公開名・test名は一覧に含めない。設定案のfieldと値はdesign.mdで定義する。既存名の改名・互換aliasは提案しない。

| 公開名の案 | 種類・配置 | 役割と既存名との差 |
| --- | --- | --- |
| `fedsda-soft-training-single-model-prediction` | 構成名・spec名 | softの責任度による学習とsingleの現在モデルによる単体予測を併記。`single-model`はモデルを1つしか保持しない意味ではない |
| `SOFT_TRAINING_SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME` | `sweep/experiment_configurations.py`の構成定数 | 新構成の名前。既存SOFT_TRAININGとSINGLE_MODEL_PREDICTIONの定数はそのまま残す |
| `runtime/training_responsibility_weight_updates.py` | 新規module案 | 実予測とは独立した学習責任度の事前重みの更新。責任度そのものを計算する`released_sample_training_responsibilities.py`とは別 |
| `update_training_responsibility_weights_after_label_observation` | 上記moduleの公開関数案 | 同期写しの損失を観測して学習用重みを進める。実際の予測の記録や帰属先を変えない |
| `recalibrate_training_responsibility_weights_after_aggregation` | 上記moduleの公開関数案 | 配布後に局所の保留損失から学習用Fixed-Shareだけを再生。既存の予測状態全体の再較正や固定済みqの再推定とは違う |
| `tools/aggregate_soft_training_single_model_prediction.py` | 新規集計module案 | 保存済み4構成の照合と対の集計。結果フォルダ内の既存Git管理外scriptと区別し、既存結果を書き換えない |
| `main` | 上記集計moduleのCLI入口 | 入出力ルートの引数処理と分析の実行。内部の集計関数を公開APIにはしない |

既存の`CurrentTrainingModelAssignment`、`FixedShareWeightController`、`SynchronizedPredictionModelSnapshot`、責任度計算・ストア・共同学習の公開名は再利用し、役割を変えない。追加の公開名が必要になった場合は実装前に本表と設計を更新して確認を受ける。D1・D2は案Aを前提とし、別案へ変更する場合は要求・設計・taskと本表を見直して仕様レビューを受ける。
