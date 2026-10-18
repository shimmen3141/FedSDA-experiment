# 命名・役割レビュー

対象spec: candidate-validation-waiting-diagnostic。公開名はCLIのmoduleのみ。

|提案名|種類|役割|既存との違い|
|---|---|---|---|
|run_candidate_validation_waiting_diagnostic|module|待機後強制採用の診断実験を実行・保存するCLI|通常掃引の構成追加ではなく、介入を明示した研究用の入口|

実装helperは全て非公開とし、既存のpost-alarm validationや通常runの公開APIを増やさない。
