# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 検査と計測
- [x] 1.1 旧の計数を、外側からの数え直しと照合するtestを書く
  - goldenの3ケースで、旧の全体runを、hookつきで実行し、旧の計数の合計と照合する。
  - 完了: 検査のtestが成功する（旧実装は変えない）。
  - _Boundary: test_legacy_computation_count_audit_
  - _Requirements: 1.1, 1.2_

- [x] 1.2 モデルの計算を、計測の区間の間、外側から数える
  - 計数の照合（手計算）、対象外のmodule、区間の後と例外の後、入れ子、差、結果と乱数を変えないことのtestを先に書く。
  - 完了: 計測のtestと、依存境界のsuiteが成功する。
  - _Boundary: model_computation_measurement_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.2_

- [x] 1.3 clientが、検出器の計算の計数を保持する
  - ownerのtestと、clientの全状態の新旧照合への、実旧の検出器の計数との照合を先に書く。
  - 完了: ownerのtest、clientのtest、依存境界のsuiteが成功する。
  - _Depends: なし_
  - _Boundary: LossMonitoringComputationCountStore、FedsdaRunClient_
  - _Requirements: 3.1, 3.2, 6.2_

- [x] 2. 重複した計算の解消
- [x] 2.1 有界損失の評価へ、3つの入口を足す
  - 既存の関数との一致、共有部を使い回した損失の一致、検査と拒否のtestを先に書く。
  - 完了: 有界損失の評価のtestが成功する。
  - _Boundary: classifier_bounded_loss_evaluation_
  - _Requirements: 4.4_

- [x] 2.2 クロス評価・警報時の区間の準備・集約後の再較正の、重複した順伝播をなくす
  - 3つの対照のtestへ、順伝播の標本数が、実旧の同じ処理の計数と一致することの確認を、先に足す（失敗を確かめる）。
  - 完了: 3つのmoduleの既存のtest（実旧との対照、拒否）と、足した確認が成功する。
  - _Depends: 1.2, 2.1_
  - _Boundary: client_model_cross_evaluation、alarm_training_interval_preparation、post_aggregation_prediction_recalibration_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 3. 計測つきの全体runと指標
- [x] 3.1 計測つきの全体runを実行し、計算量の指標を導出して、goldenの7項目と照合する
  - 指標の導出のtestを、33指標のすべてを照合する形へ、先に直す。計測つきの全体runのtest。
  - 計測つきの全体run、指標の導出の2項目を実装する。共用scriptへ足す。
  - 完了: 指標の導出のtest（Windowsでは、goldenの33指標の照合を含む）、計測つきの全体runのtest、共用script、依存境界のsuiteが成功する。
  - _Depends: 1.2, 1.3, 2.2_
  - _Boundary: fedsda_measured_run_execution、fedsda_run_metric_derivation_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.4, 5.3, 6.1, 6.2_

## Implementation Notes

- ユーザーの決定（2026-10-10）: 計算量の指標は必要。旧の計数の検査を含める。旧と同じ件数に加えて、積和演算の数を測る（逆伝播は、後から分離できる形で）。サーバの計算、ラウンドごとの系列、保有モデル数での正規化は、次のspec。
- 調査の結果: (1)旧の計数は、外側からの数え直し（PyTorchのhook）と、goldenの3ケース（sine2・sea2・mnist2）で完全に一致した（漏れ・重複なし）。恒久のtest（test_legacy_computation_count_audit.py）にした。(2)新実装は、3箇所で、旧より多く順伝播を行っていた（結果は同じ）。goldenの条件（sine2）で、概念固有部 +2057標本、共有部 +5776標本。原因は、クロス評価（+977）、警報時の区間の準備（+1080）、集約後の再較正（共有部 +3719）。
- 手順からの逸脱: 計測つきの全体run（fedsda_measured_run_execution.py）は、sourceを先に書き、その後で、専用のtestを書いた。ほかは、testを先に書いた（moduleがないだけの失敗は記録していない。重複の解消は、新旧を同じ乱数から実行するhelper `run_in_both` へ計算量の照合を足し、多数の対照が失敗することを確かめてから、sourceを変えた）。
- 実装中の発見: (1)`run_in_both`（clientの軌跡、クロス評価、配布、再較正、クラスタリング、サーバのラウンドの対照が使う）へ、「新の操作のモデルの計算（外側から数えた値）が、実旧のclientの計数の増分（共有部・概念固有部の標本数、学習の標本数、optimizerの更新回数）と一致する」照合を足した。3箇所の解消の後、全部の対照が通った。今後、部品を変えて順伝播が増減すると、この照合が検出する。(2)goldenの条件で、33指標のすべて（計算量の7項目を含む）が、実旧の実行結果と完全に一致し、Windows用のgoldenと一致した。小さい7条件でも、計算量の7項目は、実旧の集計と一致した。(3)検出器の計算の計数（更新回数、評価した候補×賭け率の数）は、監視の観測結果が、すでに返していた。clientが合計を保持するようにした。(4)積和演算の数は、標本数の計数×部品の層の形の合計と一致する（goldenの条件と、小さい条件で確かめた）。
- 設計の判断: 計測は、PyTorchの全体のhookで、実行中に外側から数える（部品へ数え方を書き込まない）。学習と推論は、順伝播のときの勾配の有無で分ける。逆伝播は、勾配つきの順伝播ごとの見積りで、順伝播とは別の項目に持つ。事前学習の計算は、準備の間の計数として別に返し、指標には含めない（旧と同じ定義）。
- 独立レビュー（2回、2026-10-10）: どちらもClaude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。GPT-6 Lunaは利用上限で使えない期間。
  - 1回め: session `4153a1bf-6099-49db-af2d-d0f32bff4e9e`、対象`f6b21ec..f4390e4`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、Minor 6件、任意 1件）。対象test（計測ほか 3169 passed、旧の計数の検査 4 passed、clientほかの対照 291 passed、指標の導出 21 passed・skipなし）、Ruff、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない。
  - 2回め（Majorの修正の確認。別session）: session `fc825cdc-fb1a-4f3a-bba8-7637e151a6ea`、対象`f4390e4..baf072c`。判定は`FIX: APPROVED`（Minor 1件——文書の表現）。対象test（3354 passed、53 passed、4 passed）とRuffを独立に実行した。足した拒否のtestが、修正の前の実装で失敗することは、レビュー担当がコードを読んで判断した（主担当も、実行しては確かめていない）。
- 指摘の採否: (1)Major: 警報時の区間の準備の検査を、順伝播なしへ変えたため、出力が不正になる分類器の拒否が、評価標本の保存（乱数の消費を含む）の後へ移り、要求4.4に反した→採用。検査を、元の「損失を計算する」（状態の更新より前）へ戻し、計算した損失を、取込みへ渡して、取込みが計算し直さない形にした（取込みへ、任意の引数`evaluated_observed_losses`を足した）。拒否のtestと、取込みのtestを足した（`baf072c`）。(2)Minor: optimizerの見分け方が、区間の中の順伝播の履歴に依存する→制約として、計測のmoduleの説明に書き、testで固定した。(3)Minor: 旧の計数の検査のoptimizerの見分け方が、パラメータの形に依存していた→順伝播した共有部が持つoptimizerとの同一性へ変えた。(4)Minor: 検査は、旧の分類層（素の全結合層）を直接は数えていない→数え方の前提を、testの説明に書いた。(5)Minor: 計測が数えない場合（入力をkeywordだけで渡す、tensorでない入力、3次元以上の入力での標本数の定義）→moduleの説明に書き、testを足した。(6)Minor: 逆伝播の見積りと、実際の逆伝播の照合は、小さい分類器だけ→設計の文面を直した。(7)Minor: 警報時の区間の準備の、実旧の計数との照合が、単体では間接的→clientの軌跡の対照（操作ごとに、実旧の計数の増分と照合する）で覆う。単体のtestは、順伝播の標本数を、直接の値で確かめる（ここへ記録）。(8)任意: 公開の検査が、非公開の検査を呼ぶ形→命名表の文面を直した（非公開の名前は、依存境界testが参照しているので、残す）。2回めのMinor（「区間の中で」という表現）: ここでの「区間」は、計測の区間を指すので、文面は変えていない。
- 検証（Windows基準環境、`baf072c`。この後は、specの文書とsteeringだけを変えた）: 全pytest 11541 passed / 3 skipped / 2 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分（federated_drift_experiment、tools、2つの旧回帰testとgolden）は空。`spec_checks.py names`は、変更・新規のsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 計測は、processの全体のhookを使う（区間の中の、すべてのモデルの計算を数える。複数のthreadからの同時の順伝播には対応しない。`forward`の直接の呼出し、keywordだけの入力、`torch.compile`ほかは、数えない）。用途別の内訳、clientごと・ラウンドごとの系列、実行時間は、ない。サーバの演算数、保有モデル数での正規化は、次のspec。逆伝播の積和演算は、見積り（実測ではない）。バイアスの加算・活性化関数・損失・optimizerの更新の演算は、数えない。照合したdatasetは、新実装ではsine2だけ（旧の計数の検査は、3ケース）。Linuxでは、このspecのtestを実行していない。
