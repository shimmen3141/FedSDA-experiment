# 設計調査と判断

## Summary

- 対象: single-run-execution。既存設定基盤からの拡張としてlight discoveryを実施。
- 根拠の詳細はreference-inventory.md。ここでは設計へ影響する判断だけを保存する。
- fable-methodとkiro-spec-designの境界・synthesis・review gateを参照した。

## Research Log

### 初期準備と乱数

- 旧experiment.pyは初期モデル構築・事前学習の後に系列と標本を生成する。
- Python shuffleと系列生成は同じ乱数列を消費し、初期標本とstream生成は同じNumPy列を消費する。
- 新実装はRandomとRandomStateをrunで所有し、初期準備にも同じ実体を渡す。default_rngへの変更は再現列を変えるため採用しない。
- 固定環境でRandomState.uniformとRandom.shuffleの利用可能性を確認した。
- CPU torch.random.fork_rngの入口でmanual_seedを実行し、例外出口で元状態と一致することをAPI probeで確認した。
- このprobeは設計上の利用可能性の確認であり、新コードのテスト成功を意味しない。

### 同期と登録可能状態

- clients/base.pyのhas_pending_modelはパラメータ存在とpending_model_readyの両方を確認する。
- 同期後のpromote_pending_to_readyはclients/fedsda.pyで待ち区間を減らす。候補存在だけを意味する命名を避けた。
- runtimeは登録可能という事実をserverへ伝え、統合方針を自分で計算しない。

### 既存境界テスト

- 設定基盤のASTテストは現在package全体を検査する。データ・モデル層追加時は検査対象を設定基盤へ限定する必要がある。
- 設定型のNumPy/torch禁止を維持し、データ・CPU乱数境界の許可依存は別途検証する。

## Architecture Pattern Evaluation

| 案 | 判断 |
|---|---|
| 旧関数をproduction wrapperから呼ぶ | 旧config依存と旧名が実行経路へ残るため不採用 |
| 全手法を一括移植 | 初回の検証単位を超えるため不採用 |
| 純粋なデータ供給と、狭い処理部Protocolを持つ進行制御 | 採用。初期準備後の列と処理順を一つのrunで独立検証できる |

## Design Decisions

- 標準dataclass・Protocol・既存乱数APIを使い、新しいDI frameworkやイベント配信基盤を導入しない。
- 標本・記録はtupleで不変にする。共有する可変乱数実体と参加者は実行中の所有物で、成功結果へ含めない。
- 生成規則と区間進行は別ファイルでテストできるが、初回の受け入れは初期準備から終端までの単一経路。
- seed上限は実行方式の制約として拒否し、既存設定型の整数を丸めたり既存coreの値域を変更しない。
- 真の概念を標本処理Protocolへ渡さず、結果の評価用列として保持する。
- 同一プロセス内の並行runは対象外。並列実験は後続のprocess単位の実行層で扱う。
- 草案のレビューで、Protocolの引数型をruntimeへ置くと逆依存になることを確認した。run乱数源の型と生成をexecutionへ配置し、runtimeはそれを使う。データ関数は個別の標準乱数型だけを受け取る。
