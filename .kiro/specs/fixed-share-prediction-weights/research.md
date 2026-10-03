# 調査と設計判断

## Summary

既存コードの移植なのでlight discoveryを適用。旧client全体では予測・割当・監視・候補・学習が結合するため、stdlibで独立する予測重みを最初に移植する。
適用: fable-method、kiro-spec-init、kiro-spec-requirements、kiro-spec-design。主担当が調査を統合して境界を決めた。

## Research Log

参照: `federated_drift_experiment/expert_routing.py:384–520`、`tests/test_expert_routing.py`。

- ID昇順、一様初期化、集合差だけで重み・分散消去、初回を除く集合変更計数。
- 予測時点の重みで期待損失・分散を計算。学習率、指数更新、一様再配分の演算順を維持する。
- 単一モデルは重み1。計数のleaderは最小ID、公開選択だけ同率優先IDを許す。
- 空再生は全状態不変。非空は証拠だけ消去し、途中の集合変更も許す。
- 旧updateの不正入力による部分変更は新境界で修正する。正常入力のアルゴリズムは変えない。

後続調査: `clients/fedsda.py`、`clients/shared_backbone.py`、`e_detector.py`、`tests/test_e_detector.py`、`tests/test_provisional_model.py`。

- 出力混合: torch float32のbinary閾値・softmaxを保持する別責務。
- ClassESR: 現行学習モデルの損失、overall/正解class/他classの直近値、混合閾値、遅延class初観測時baseline、内部時刻とglobal位置を所有する。
- 候補: 将来損失による現行モデル優先再適合、前後半の厳密改善。torch平均、奇数件、等号境界を保持する。
- 割当: FIFO帰属、統計、flush、候補登録・ID採番を調整する。予測leaderと学習割当先を分ける。

## Architecture Pattern Evaluation

| 案 | 判断 |
|---|---|
| client一括移植 | 状態所有と検証範囲が広すぎるため見送る |
| 各演算にinterface/adapter | 単一数値部品には過剰なため見送る |
| 機能別単一controller | 採用。固定設定へだけ依存する |

## Design Decisions

- Build vs Adopt: 旧基準の演算順保持が目的なので数値処理を新APIへ移植し、新ライブラリを追加しない。
- Simplification: 重みは独立dictで返す。snapshot型・routing alias・共通frameworkは不要。
- 入力検査の総和だけmath.fsumと絶対許容差1e-12。受理値を再正規化せず、計算本体の通常sumを保つ。
- 確率0は有効。有限0～1・総和1を要求する。
- 全再生列を検証・コピーしてから変更するため、列長に比例する検証メモリを使う。
- 極端に大きい整数時間尺度のfloat変換失敗はconstructorで拒否し、状態変更途中のOverflowErrorを避ける。既存設定型の値域自体は変更しない。

## Risks & Mitigations

浮動演算順は各操作後の旧oracle直接照合、依存は新src全走査と禁止例注入で検出する。既存golden保持、部品同値性、新全体runの完成を別々に記録する。
