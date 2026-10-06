# 調査根拠

- 旧BaseClient.confirm_model_registration（clients/base.py:425）は負のcurrent_model_idについて`models[new_global_id] = models.pop(temp_id)`を実行する。実体内のoptimizer/parameter/共有部はそのまま。既存先位置維持、未登録先/同ID末尾。
- 元モデルが欠落しpending parameterがある場合の旧再生成は、上位の別責務。実旧oracleではpending=None/他owner空を明示してregistry部分だけを照合する。
- held-model-training-state-registryはID対応を後続へ分離済み。recordはfrozenでIDを持ち、NNとParameterOptimizerStateはlive参照。ID変更時に新recordを作り、古いrecordのIDは変更しない。
- optimizerはrecordへ固定しない。新bindingは既存snapshot APIで管理器の現在optimizerを取得する。ID付替え自体はresetしない。
- loss-statistics-model-id-reassignmentの統計上書きと接続できるが、上位呼出しだけをtest-onlyで確認する。他ownerをregistryへ取り込まない。
- 旧の非負current_model_id早期returnは上位の正式登録条件。汎用registryの正元IDは直接期待値で確認し、実旧負ID対照と区別する。
- 現時点で新たな旧実装不具合は観測していない。
