# Research & Design Decisions

## Summary
- Feature: feddrift-run-client-assembly。Discovery Scope: 既存部品の接続。
- 初期cloneと観測/round/終端を一clientへまとめる。数値・判定は移植済み部品を使う。

## Research Log
### 実旧clientと実行順
参照: `federated_drift_experiment/clients/base.py`の`__init__`、`clients/feddrift.py`、`experiment.py`の`_run_feddrift_timestep`。旧生成はinitial_modelsをdeepcopy、現在ID0、統計clone、全store空、ID開始値-100-client ID、前回損失None。実行は全clientのprocess_batch→server.prepare_timestep→R回の全client.local_train(k_steps=検出batch件数)/server round。既存FedSDAループのflush→同期の形へ流用すると、Rと準備の順序が失われる。

### Oracle実行の事前確認
実旧__init__/空process_batch/local_train(0)/flushを実行するscriptを`venv/refactoring-tests/feddrift-run-client-assembly-oracle-preflight.py`へ置いた。CLIの引用符とPYTHONPATHを補正後にPASSを確認した。実旧を代役へ替えず、既存build_modelsの同一parameter対照を再利用する。対照testで実旧のdeepcopyがparameter.gradをNoneにすることを確認し、初期勾配も保持するという当初の案を要求/設計で補正した。入力のgradは変更しない。

### 設計の合成
参照: `runtime/fedsda_run_client.py`の初期clone、FedDrift batch/stream/round、既存ownerの型検査。新たな状態機械や汎用Protocolを足さず、具体clientから既存部品へ委譲する。初期classifier+optimizerの同時deepcopyで参照とAdam状態を保つ。RNGを消費するモデル再生成は行わない。標本変換だけを入口に足し、全blockの検査はstreamへ接続する。

## Design Decisions
FedSDAへの共通化は全体進行の違いが大きく採用しない。clientの設定束は必要な条件だけを置く暫定形で、完成run設定のspecで置き場所を決める。初期モデル0のみは実experimentの入力に対応する。任意の初期モデル集合や共有モデル構造は今回の契約に含めない。

## Risks and Mitigations
旧new model生成や評価抽出はRNGを消費する。両経路の開始時3RNGを揃え、旧はglobal Python RNG、新はRandomへ明示的に対応付ける。全体run・server/goldenの一致はこの部品対照だけでは主張しない。

## Consulted Skills
kiro init→requirements（EARS/review gate）→design（light discovery/synthesis/review gate）→tasks（task graph gate）→impl --review off/TDD→verify completion→validate impl。fable-methodにより根拠の確認を前倒しし、子エージェント禁止のため全調査・段階gateを主担当inline、最後の独立レビューを別CLIとする。
