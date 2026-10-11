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
- task 1: 対照testを先に書き、未実装関数がNotImplementedErrorで失敗することを確認した（1 failed）。実装後の対象testは62 passed。task 2の依存登録ではexact検査対象への登録が1か所不足し、追加して対象と依存境界のtestが3108 passedになった。Ruff・Pyright成功。独立レビューは未実施（以下の実行許可待ち）。

- Windows全回帰: `40083da`で12647 passed / 4 skipped / 4 warnings、exit 0（681.26秒。`-n 8 --dist loadfile`）。JUnitは`venv/refactoring-tests/feddrift-detection-batch-assessment-full.xml`、実行ログは同じstemの`.log`。前specの12582 passedに、新規判定62件と依存拒否3件を足した件数と一致。旧11ケース・最終3ケースのgoldenと、既存構成の実旧照合、共用fresh process smoke（新しいFedDriftの呼出しを含む）が成功。Ruff check/format、Pyright 0 errors、pip check成功。固定旧748c3aaの差分が空。source hashは`f7b0446ccbf16aa92939352c7ba764662c1acc4ba363269771e6f1e05133c5e3`（401パス）。
- 独立レビューは起動していない。Haiku 5.5・effort mediumの独立CLIを、数値比較の移植のレビュー用に選び、対象40083daと許可した検証だけを依頼する準備をしたが、自動承認レビューが外部送信の承認不足で拒否した。依頼文は`venv/refactoring-tests/feddrift-detection-batch-assessment-review-prompt.txt`。ユーザーへ対象ソース・test・specと必要な旧対照コードをAnthropicのClaude CLIへ渡すことの許可を依頼済み。自己レビューや別の外部CLIで拒否を迂回せず、task 2と命名承認を未完了のままにする。
- 手順との差: 独立レビューの起動が拒否されたため、影響しない全回帰を先に実行した。指摘の反映でsourceを変更した場合は、反映後に全回帰を取り直す。WSLと変異テストは実行していない。FedDriftのclient・サーバ・全体runとgoldenはまだ移植していない。
