# Implementation Plan

- [ ] 1. 2つの扱いの計算を足す（接続はまだしない）
- [x] 1.1 学習者の予測的符号長と学習者の選択
  - warm startの学習者の予測的符号長（runの乱数を変えない、現在のモデルを変えない）、学習者の選択（短い方、同じ値の順、ln Kを足す）、選択の結果の学習者の名前、のtestを先に書き、REDを確かめる。既存の関数の改名（`compute_new_model_prequential_code_length`→`compute_reinitialized_learner_prequential_code_length`、`derive_new_model_initialization_seed`→`derive_reinitialized_learner_initialization_seed`。呼出し・test・依存境界・研究ツール）を行い、値が切り出しの前と同じことを確かめるtestを含める。
  - 完了条件: 新しいtestと、最小記述長の選択の既存のtestが通る。依存境界の許可を登録する。
  - _Requirements: 1.2, 1.3, 1.5_
- [ ] 1.2 初期モデルの統計の交差検証
  - 組の分け方と足す順、runの乱数を変えない、初期モデルを変えない、のtestを先に書き、REDを確かめる。`pretrain_initial_model`の切り出しの後も、既存のtest（実旧との照合を含む）が通ることを確かめる。
  - 完了条件: 新しいtestと、事前学習・参加者の準備の既存のtestが通る。
  - _Requirements: 2.2_

- [ ] 2. 設定を足し、接続する（既定は、まだ変えない）
  - 設定の検査（2つの項目の値、選び方との組合せの拒否、標本の数の拒否）と、接続のtest（今の扱いでは今と同じで、warm startの学習者と一時的な分類器を作らない。新しい扱いでは、選択の記録に学習者の名前が残り、統計が替わる）、計算回数のtestを先に書き、REDを確かめる。
  - 設定の項目と検査、clientと参加者の準備からの受け渡し、選択の記録の列。全部の設定の組立てで今の扱いを明示し、旧と比べる入口と、基準版1〜10の「後に足された項目」の表へ足し、版1〜10の比べる形から新しい列を取り除く。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜10）が値を変えずに通る。
  - _Requirements: 1.1, 1.4, 1.5, 1.6, 2.1, 2.3, 2.4, 2.5, 4.1, 4.2, 5.4, 5.5_

- [ ] 3. 研究ツールと同じ値になることを確かめる
  - 研究ツールに、2つの新しい扱いを設定で選ぶだけの変種を足す。照合のscriptで、手元の条件（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0、2500標本）で、組み合わせの変種と、この変種の、精度・通信量・予測の記録のdigest・最終のパラメータのdigestが一致し、計算回数は、学習の計数が一致し、推論の計数の差が、研究ツールだけが行う、各組の一時的な分類器での学習に使った標本の評価（組ごとに学習に使った標本を1件ずつ1回の順伝播。5組で事前学習の標本の数の4倍）の分だけであることを確かめる。
  - 完了条件: 一致の結果（条件ごと）と、照合のscriptの場所がImplementation Notesにある。一致しない場合は、原因を特定して直すまで次へ進まない。
  - _Requirements: 6.1_

- [ ] 4. 最終構成の既定を替え、研究ツールの基点を固定する
  - 既定と全FedSDA構成の値のtest（候補検証ありの構成は今の扱い）を先に書き、REDを確かめてから、既定を替える。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。
  - 研究ツールの全部の変種で、2つの項目を基準版10の値に戻してから差し替えを行い、製品の既定をそのまま使う変種を足す。基準版10の既定の変種（`default`）の値が、既定を替える前と同じことを、手元の小さい条件で確かめる。
  - 完了条件: 構成のtestが通り、基準版1〜10と実旧との照合が値を変えずに通る。研究ツールの`default`の値が、替える前と同じ。
  - _Requirements: 3.1, 6.2_

- [ ] 5. 基準版11を足す
  - 実装の前の確認: 新しい既定の大きい条件の2ケースで、warm startの学習者が短かった「新しいモデル」の選択が1回以上起きるかを確かめる（起きなければ、ケースの選び直しを決める）。
  - 版11の定義、版10→11の差の説明のtest、必須の経路を足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を11へ上げる。
  - 完了条件: 版1〜11の照合と、版10→11の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 6. 文書を合わせる
  - 手法の説明、基準（版11）、実験の入口、研究記録（NEW-006・LEGACY-020の状態）、再開案内。
  - 完了条件: 文書が2つの新しい扱いを説明する。
  - _Requirements: 7.1_

- [ ] 7. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 背景（2026-10-22）: 研究ツールの組み合わせの変種`default-two-new-model-learners-cross-validated-initial-model-statistics`（commit `791cce4`）の本規模（uranium、240 run、MNIST4のseed 10〜19の40 run）。結果と調査は[NEW-006](../../../docs/research/implementation-findings/new-006-minimum-description-length-keeps-partially-changed-concept-in-current-model.md)。ユーザーが既定にすると決めた。
- 仕様レビュー1回目（Luna high、session `01a1628c-c9ea-7981-91c3-51b0307d2f22`）: CHANGES REQUIRED。Major 1件（研究ツールは各組で`pretrain_initial_model`を呼び、学習に使った標本の評価——捨てる統計——も行うので、製品で省くと計算回数が一致しない）。製品では評価を省く方針を保ち、照合の条件を「学習の計数が一致し、推論の計数の差がその評価の分だけ」に直した（要求6.1、設計、Task 3）。
- 命名の変更（2026-10-22、ユーザー指摘）: warm startの学習者の関数と並ぶのに、既存の関数が仮説（new model）の名前だったので、`compute_reinitialized_learner_prequential_code_length`・`derive_reinitialized_learner_initialization_seed`へ改名することにした。公開名の変更なので、仕様レビューで確認を受ける。
- 仕様レビュー2回目（Luna high、session `01a1628f-0ff5-76e0-ab1b-3bd43edcb25f`）: PASS（1回目の修正と、計算回数の差の見積り——研究ツールの方が4N回の順伝播だけ多い——を確認）。
- 仕様レビュー3回目（改名の部分、Luna high、session `01a16292-67ab-7ba2-b24d-f8444ea88384`）: Minor 1件（改名するseedの関数の定義場所が、別のmoduleの節に書かれていた）。定義場所を書き足した。改名後の名前の役割と影響範囲は問題なしとの判定。
- Task 1.1: REDは、新しい名前が無いことによるtestの収集の失敗。2つの関数を改名し（src・test・依存境界・研究ツール）、標本を順に予測して学ぶ部分を`_accumulate_prequential_code_length`へ切り出し、`compute_warm_started_learner_prequential_code_length`、`select_new_model_learner`・`NewModelLearnerSelection`、選択の結果の`new_model_learner_name`を足した。保留の列での選択は、学習者の組の設定（Task 2）が入るまで、初期化し直した学習者だけを`select_new_model_learner`へ渡す（足す値は0）。純粋な計算と保留の列での選択のtest 48件、依存境界と最小記述長の接続のtestを合わせて3181 passed。版10のgoldenの大きい2ケース（SINE2・MNIST2の採用の構成）の照合12件が通り、切り出しと改名で値が変わらないことを確かめた。Ruff・Pyright 0 errors。
