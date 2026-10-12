# Design Document

## Overview
既存のcross_evaluate_global_modelsの呼出し内の集合をモデル構造によらず利用する。公開API・設定・成果物形式を追加しない。

## Boundary Commitments
### This Spec Owns
新FedSDAのクロス評価の転送計数と、この意図した旧との差のテスト側対応付け。
### Out of Boundary
旧source/goldenの修正、他のLEGACY、全構成goldenの生成、FedDrift方式の変更。
### Allowed Dependencies
既存CommunicationVolumeRecordStore、パラメータ分割、クロス評価記録。sourceのimport集合は維持する。旧APIの観測はtest内だけ。
### Revalidation Triggers
下り通信量を利用する指標・保存と独立構成の全体run照合を再検証する。共有部ありのクロス評価、FedDrift、旧11/最終3goldenは全pytestで確認する。

## Architecture
### Existing Architecture Analysis
共有部が空のときだけ、現在は集合の確認を飛ばして毎評価で全パラメータを送る。LEGACY-018のために維持していた分岐である。共有部がある場合の既存集合は呼出し内にあり、候補とclientの組を保持する。
### Components and Interfaces
runtime/model_cross_evaluation.pyのcross_evaluate_global_modelsは空の共有部でも概念固有部の既存集合を使う。共有部は空なら記録しない。モデル数1と概念固有部の転送は未送信の組でだけ足す。評価処理とメッセージの位置を動かさない。実際のネットワークキャッシュは追加しない。

## File Structure Plan
- src/federated_learning_experiments/runtime/model_cross_evaluation.py: 重複計上分岐の除去。
- tests/refactoring/test_cross_evaluation_download_deduplication.py: 呼出し単位の計数、実旧転送の観測helper、独立構成の保存検証。
- tests/refactoring/test_fedsda_run_metric_derivation.py: 実旧を変更せず転送イベントを観測し、既存小規模全体runの通信量を明示的に補正した期待値へ照合。
- tests/refactoring/test_server_model_registration_and_aggregation.py: 既存全状態照合へ、観測済みの旧転送重複を減算した通信量の期待値を接続。
- tests/refactoring/fresh_process_smoke.py: 新実装だけの独立構成の全体runと保存。
- tests/refactoring/test_fedsda_configuration_variants.py: 独立構成の合成/MNIST全体照合と重複経路の観測を必要な範囲で調整。
- docs/research/implementation-findings/legacy-018-cross-evaluation-download-volume-without-shared-part.md、README.mdと.kiro/steering/resume.md: 修正状態と次の一手。

## System Flows
評価先抽出→組ごとの依頼/返信計数→clientの共有部未送信なら転送→候補/client未送信なら概念固有部を転送→従来どおり評価と記録。集合はこの呼出しを越えない。途中失敗のrollbackは追加しない。

## Validation Design
部品の再現テストは実体型clientとrepositoryを使い、候補2個/client複数、複数保有、client上限による抽出、評価先無し、2回の呼出しを検証する。共有部ありの既存テストを再利用する。
全体runでは旧CrossEvaluationClusteringServerのbegin/recordの呼出しをtest側wrapperで観測する。元のメソッドを必ず呼び旧の計数・RNG・評価を変えない。旧_record_cross_evaluation_model_transferの署名は(self, model_id, params, target_clients)であり、_cross_evaluateは候補の正式model_idを渡す。重複のキーは(model_id, client.client_id)であり、パラメータの値やobject identityでは判定しない。同じパラメータ値の別ID候補も別転送として検証する。各beginで送信済み組をresetし、recordのmodel_id・実paramsとtarget_clientsから重複モデル数、numel、numel*element_sizeを積算する。新の下り3項目は旧の最終値からこの重複を引いた値へassertし、他5項目は旧との等値を維持する。モデル構造が独立の場合のみ適用し、一般的な比較除外flagを追加しない。元の全状態照合と離散列比較を維持する。保存は新のJSON/NPZで通信量がrun結果と一致することを確認する。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | 呼出し内の既存集合、部品再現と再呼出し |
| 2 | 評価/抽出/メッセージ順維持、既存全状態対照と拒否/失敗test |
| 3 | testだけの旧転送イベント観測、全体run補正期待値、保存と全回帰 |
