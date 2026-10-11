# Implementation Plan

- [x] 1. 検出バッチの評価と判断を実旧に照合して実装する
  - 二値・多クラスの実旧モデルを使った対照testを先に書く。
  - 保有順の評価、最小値と同率選択、初回と閾値境界、部分バッチを照合する。
  - 不正入力のforward前拒否と、パラメータ・既存勾配・学習モード・乱数の保持を確認する。
  - 全入力の検査後にだけ評価する不変な結果が得られ、対象testが成功することを完了条件とする。
  - _Boundary: 検出バッチの数値評価と比較判断_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2_

- [ ] 2. 依存境界と新実装だけの実行へ接続し、回帰と独立レビューを通す
  - 新moduleの許可依存を登録し、上位runtimeへの依存を拒否する。
  - 共用fresh smokeへ、実分類器を使った判定を足す。
  - 全taskの独立レビュー1回の有用な指摘を反映し、固定Windows環境の全pytestと品質検査が成功することを完了条件とする。
  - _Boundary: 判定からlearningへの接続と検証_
  - _Depends: 1_
  - _Requirements: 3.3_

## Implementation Notes

- 要求・設計のreview gateとtask-graph sanity: PASS（主担当inline。共通引継ぎ手順により子エージェントは使わない）。既存の固定環境以外に前提はなく、全9要求を上の2taskへ割り当てた。
- 実装前に文書を作成した。数値境界だけを切り出す理由はresearch.md。後続clientと全体runの検証を本specの完了証拠に含めない。
- task 1: 対照testを先に書き、未実装関数がNotImplementedErrorで失敗することを確認した（1 failed）。実装後の対象testは62 passed。task 2の依存登録ではexact検査対象への登録が1か所不足し、追加して対象と依存境界のtestが3108 passedになった。Ruff・Pyright成功。全回帰と独立レビューは未実施。
