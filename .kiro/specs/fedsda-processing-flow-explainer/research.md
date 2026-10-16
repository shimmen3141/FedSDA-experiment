# Research & Design Decisions

## Summary
- 対象: `3e6153c`。元HTMLは2026-10-02版、31908bytes、Git管理外。専用worktree/branchで既存のClaude作業と分離した。
- 適用skill: fable-method、kiro-spec-init/requirements/design/tasks。要件gate・設計gate・task sanityの観点で下書きを確認。内蔵子エージェントは共有規約に従い不使用。
- 外部依存や新しいフレームワークは不要。ローカル実装を根拠にし、研究結果の数値は説明HTMLへ載せない。

## Research Log
### 旧図の読みづらさ
入力の複製、A/Hや損失の無説明、1枚のSVGに予測・ストア・FIFO・検証・serverを混在、遠い機構への矢印、採否分岐の同時表示、コマ移動の循環とキー横取り、実装対応が旧API、TとFIFOの同一視。単なる座標修正では責務と時間順が整理できない。

### 現行の根拠（すべてsrc/federated_learning_experiments配下）
| 根拠 | 確認内容 |
|---|---|
| `runtime/observed_sample_processing.py` | 予測→保持中検証→局所現行損失監視→FIFO追加→不足時の警報待ち→警報/非警報処理 |
| `runtime/observed_sample_prediction.py` | 混合の二段、モデル別重みは同期側損失、出力/正否記録後に重み更新 |
| `methods/fedsda/prediction_combination/synchronized_and_local_prediction_mixture.py` | 確率の結合、各結合確率の1−正解確率、二者の重みの持続 |
| `methods/fedsda/prediction_combination/synchronized_prediction_model_snapshot.py` | 配布後の全分類器の写し、共有オブジェクト維持、新規モデルのfallback |
| `methods/fedsda/model_registration/pending_model_upload.py` | 現行の保留モデルだけ待ちが進む登録方針 |
| `runtime/fedsda_run_server.py` / `runtime/server_round_synchronization.py` | 新規登録可能フラグ→統合有効、登録/集約/評価/統合/配布/再較正 |
| `runtime/fedsda_final_configuration_run_settings.py` | 既定は局所パラメータ、現行として待つ登録、警報を最小件数まで待つ |
| `runtime/post_alarm_candidate_validation_progress.py` / `runtime/held_candidate_validation_progress.py` | 検証観測と確定、候補の継続学習呼出しはない（初期学習後は検証中固定） |

関連spec: fedsda-synchronized-local-mixture-prediction、fedsda-fixed-share-time-scale-axis、fedsda-default-waited-registration-and-alarm-wait、pending-model-registration-after-waiting-as-current-model。ユーザーはFIFO容量とTの独立、Tの二用途での共用、設定数値を出さないことを指定した。

## Architecture Pattern Evaluation
一枚に全矢印を出す案は、モデルを中心に多数の受渡しが交差するため不採用。モデルを用途ごとに無説明に複製する案も別実体と誤解される。全体の構造図＋章ごとの小さな直列図を採用し、同期の写しだけは別のパラメータ状態として明示する。

## Risks & Mitigations
- 最新の実験構成とコード既定が異なる: 初期表示は混合構成、既定は局所と明記し切替可能。
- コードが並行して変わる: 基準commit固定、完了前にClaude側の関連差分を確認。
- HTMLは保留資料: 元をバックアップし、最終反映前にsha256照合。stageしない。
- 説明の省略が仕様化される: 写しの例外、検証の順、登録の待ち、再較正で保持する二者重みを補足へ残す。

仕様レビュー初回のMajor3件を受け、警報/非警報の排他的な経路と標本行き先、検証の判定段階、同期の実行条件を明示した。追加の主担当調査で候補が検証中に追加学習しないことを確認し、下書きの説明を訂正した。
