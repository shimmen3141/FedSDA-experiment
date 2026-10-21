# Implementation Plan

- [x] 1. 最小記述長の選択の計算を足す（接続はまだしない）
  - 純粋な計算（仮説の符号長と選択、切り替わりの符号長、選び方の値の検査）と、モデルを使う計算（対数損失、予測的符号長、seedの導出、保留の列での選択）のtestを先に書き、REDを確かめる。
  - 完了条件: 新しい部品のtestが通る（3種類の選択、同じ長さの順、刻みより短い列、runの乱数を変えないこと、同じseedで同じ値、共有部を更新しないこと）。依存境界の許可を登録する。
  - _Requirements: 1.2, 1.5, 1.6_

- [ ] 2. 設定を足し、標本の処理と警報の処理へ接続する（既定は、まだ変えない）
  - 設定の検査（候補検証との組合せの拒否を含む）と、接続のtest（今の扱いでは今と同じ、新しい扱いでは選択の長さと行き先が使われる）を先に書き、REDを確かめる。
  - 設定の項目と検査、clientからの受け渡し、`process_observed_sample`での選択と変化区間の長さの置き換え、警報の処理への受け渡しと、`resolve_alarm_change_interval`での行き先の使い方。全部の設定の組立てで今の扱いを明示し、旧と比べる入口と、基準版1〜9の「後に足された項目」の表へ足す。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜9）が値を変えずに通る。
  - 選択の記録の置き場と記録の表を足し、版1〜9の比べる形から、この表を取り除く（今の扱いでは空の表）。今の扱いで選択の計算が呼ばれないことのtestを足す。
  - _Requirements: 1.1, 1.3, 1.4, 1.7, 1.8, 3.1, 3.2, 4.4_

- [ ] 3. 研究ツールと同じ値になることを確かめる
  - 研究ツールの変種のseedを、製品と同じ式に揃える。照合のscriptで、手元の条件（MNIST2・SINE2・CIRCLE2・SEA4、間隔100、seed 0、2500標本）で、研究ツールの変種と、新しい扱いを選んだ製品の全体runの、精度・通信量・予測の記録のdigest・最終のパラメータのdigest・計算回数が一致することを確かめる。
  - 完了条件: 一致の結果（条件ごと）と、照合のscriptの場所がImplementation Notesにある。一致しない場合は、原因を特定して直すまで次へ進まない。
  - _Requirements: 5.1, 5.2_

- [ ] 4. 最終構成の既定を替える
  - 既定と全FedSDA構成の値のtest（候補検証ありの構成は今の扱い）を先に書き、REDを確かめてから、既定を替える。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。
  - 完了条件: 構成のtestが通り、基準版1〜9と実旧との照合が値を変えずに通る。
  - _Requirements: 2.1_

- [ ] 5. 基準版10を足す
  - 実装の前の確認: 新しい既定の大きい条件の2ケースで、警報の区間の解決の3種類が、それぞれ1回以上起きるかを確かめる（起きなければ、ケースの選び直しを決める）。
  - 版10の定義、版9→10の差の説明のtest、必須の経路を足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を10へ上げる。
  - 完了条件: 版1〜10の照合と、版9→10の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 6. 文書を合わせる
  - 手法の説明、基準（版10）、実験の入口、研究記録、再開案内。
  - 完了条件: 文書が新しい扱いと、サーバのクラスタリングの閾値が残ることを説明する。
  - _Requirements: 6.1_

- [ ] 7. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 背景（2026-10-21）: 研究ツールの変種`default-detector-gated-segment-assignment`（commit `080f6bc`）の本規模の10 seed（uranium、240 run）で、24条件のうち17で既定との差が0以上、崩れるrunなし、下りは0.84〜1.24倍（[設計と結果の8〜10](../../../docs/research/fedsda-unified-concept-assignment-inference.md)）。ユーザーが既定にすると決めた。
- 仕様レビュー1回目（Luna high、session `01a15c25-47e8-7592-9e2e-52346797d0fe`）: CHANGES REQUIRED。Major 2件（必須の経路を、下流の指標——現在のモデルの維持ほか——で数えると、選んだ仮説を確かめられない。今の扱いで計算回数が変わらないことの確かめ方が無い）、Minor 1件（seedの導出の文字列の形式）。選択の記録を足して選んだ仮説で数える、今の扱いで選択の計算が呼ばれないことのtestと版1〜9の照合（計算回数を含む）で確かめる、`:`で区切る形式に決めてtestで固定する、と直した。
- 仕様レビュー2回目（Luna high、session `01a15c28-2d99-7313-9e1f-b490f1d1d390`）: 1回目の3件は解消。Major 1件（版9→10の差の説明で、新しい記録の表の比べ方が未定義）。両方の版を実行した結果を比べ、版10の結果から表を取り除いて比べ、取り除いた表が空であることを確かめる、`coverage`も版10で足す項目を取り除いて比べる、と直した。
- 仕様レビュー3回目（Luna high、session `01a15c29-7760-79a3-a225-6c66be918cfa`）: 設計は解消。Major 1件（要求4.2が、設計の除外を書いていない）。要求4.2に、両方の版を実行して比べること、表と経路の記録の除外、取り除いた表が空であることを書いた。
- 仕様レビュー4回目（Luna high、session `01a15c2a-6e59-71a3-b854-6c0e8e423d63`）: PASS。
- Task 1の途中の設計の変更（2026-10-21）: 依存境界の規則（runtimeの層はtorchを直接importしない。learningの層もmoduleごとに許可した名前だけ）に合わせ、`evaluate_classifier_per_sample_log_losses`を`learning/prediction/classifier_log_loss_evaluation.py`へ、`compute_new_model_prequential_code_length`を`learning/training/prequential_code_length.py`へ置いた（設計では`runtime/alarm_interval_minimum_description_length_selection.py`だった）。初期化の乱数は、既存の`isolated_cpu_torch_random_state`を使う。公開名の置き場所の変更なので、変更部分の仕様の再確認を受ける。
- 仕様の変更部分の再確認（Luna high、session `01a15c3b-0331-7ea3-8185-d77acb81cbd9`）: PASS（置き場所と依存の方向、命名表との一致、CPUの初期化の乱数が研究ツールと同じになること）。
- Task 1: REDは、新しいmoduleが無いことによる収集の失敗（新しいmoduleだけのtask）。純粋な計算のtest 17件、モデルを使う計算のtest 12件、依存境界のtestと合わせて3156 passed。Ruff・Pyright（`--pythonpath`付き）0 errors。純粋な計算のtestの1件は、最初、testの例の作り方の誤り（保有モデルの前半の損失を0にしたため、区間10が最短になった）で失敗し、例を直した。`reset_parameters`の呼び出しは、Pyrightの型の推定に合わせて`getattr`で取り出して呼べるときだけ呼ぶ形にした（torchの部品では同じ意味）。
