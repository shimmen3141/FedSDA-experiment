# FedSDAの既存モデル再利用

> 文書の役割: 個別機能の実装資料
>
> 対象と採用状況: 警報直後の既存モデル再利用を対象とする。2026-10-21から、新実装の最終提案の既定は、最小記述長の原理による選択（下の節）。固定の許容量による最良適合モデル選択は、新実装の設定の選択肢と旧実装に残る。警報後の候補検証は新実装から削除した。
>
> 最終構成の正本: [FedSDAの最終提案構成](../overview/proposed-method.md)

## 固定の許容量による最良適合モデル選択

新実装の設定値は`alarm_interval_assignment_policy=detector_change_span_and_mean_loss_increase_tolerance`。

FedSDAはドリフトを検出すると、推定した新概念区間のデータで保持モデルを評価する。
各モデルについて、区間平均損失とそのモデルの履歴平均損失との差が
`FEDSDA_DISTANCE_THRESHOLD`以下なら、現在の概念へ適合する既存モデルとみなす。

適合するモデルが複数ある場合は、区間平均損失が最小のモデルを再利用する。
この選択は固定動作であり、設定パラメータは持たない。

現行モデルが適合する限り維持する方式も比較したが、MNIST2では正しいモデル切替まで
抑制し、precision、recall、F1、accuracyのいずれも改善しなかったため削除した。

Pareto CSVには診断用として次の件数を保存する。

- `model_reuse_current_fit_count`: 損失最小として現行モデルを選んだ件数
- `model_reuse_alternative_fit_count`: 損失最小として別モデルを選んだ件数

これらは選択方式ではなく、固定された最良適合モデル選択がどのモデルを選んだかを表す。
旧実装の前向き再適合判定の説明は[new-model-creation.md](new-model-creation.md)に残す。新実装は、保持モデルを使えなければ候補をその場で採用する。

## 最小記述長の原理による選択（既定）

新実装の設定値は`alarm_interval_assignment_policy=minimum_description_length`（基準版10。[spec](../../.kiro/specs/fedsda-minimum-description-length-alarm-interval-assignment/design.md)、経緯は[研究記録](../research/fedsda-unified-concept-assignment-inference.md)）。警報を出す時点は、これまでどおりClassESRが決める。警報の後の変化点と帰属先を、固定の許容量の代わりに、最小記述長（MDL）の原理で選ぶ。

1. 列W＝帰属の保留（FIFO）の標本と警報の標本（古い順、最大31件）。警報の標本は、容量を超えた分を解放する前に保留へ足すので、Wは容量（既定30件）＋1件になり得る（`pending_training_assignment_buffer.py`の`append_observed_sample_index`。選択でも切り詰めない）。各保持モデルで、標本ごとの対数損失−ln p(真のクラス)を計算する。
2. 次の仮説の符号長（nats）を比べる。
   - 現在のモデルのまま: Wの全部を現在のモデルの対数損失で符号化する。
   - 変化点τで保持モデルmへ: τより前を現在のモデル、後を保持モデルm（損失の統計を持つ、現在のモデル以外のモデル）で符号化し、切り替わりの符号長を足す。
   - 変化点τで新しいモデルへ: τより前を現在のモデル、後を新しいモデルの予測的符号長（prequential code length）で符号化し、切り替わりの符号長を足す。予測的符号長は、学習者を共有部を固定して、標本ごとに「予測して対数損失を足し、1回更新する」を繰り返した和。学習者は2つ（2026-10-22から。基準版11、`new_model_hypothesis_learners=reinitialized_and_warm_started_learners`）: 現在のモデルの写しの概念固有部（adapterと出力層）を初期化し直した学習者（既定の初期化seedは、clientの番号と警報の標本の番号から導き、変化区間の長さkには依存しない。runの乱数は変えない）と、現在のモデルの写しを初期化せずに学ばせるwarm startの学習者。短い方に、どちらの学習者かを表す符号長ln 2を足す（同じ値なら初期化し直した学習者）。一部のクラスだけが変わる概念（MNIST4）はwarm startの学習者が、全体が変わる概念（SINE2）は初期化し直した学習者が短く説明する。学習者は符号長を測るためだけに作って捨て、実際に作るモデル（候補）とは別。基準版10までの扱い（初期化し直した学習者だけ）は`reinitialized_learner`。
3. 切り替わりの符号長は−ln(1/T)＋ln(行き先の数＋1)。Tは予測のFixed-Shareと同じ時間尺度（既定120）。変化区間の長さ（Wの末尾k件）の候補は、基準版16から、最小変化区間（既定5標本）以上・Wの長さ以下の全部のkを1件刻みで比べる。
4. 最短の仮説を選ぶ（同じ長さなら、現在のモデルのまま、短い区間、保持モデル、新しいモデルの順を優先）。選んだ変化区間の長さを、検出器の推定の代わりに、短い警報の判定・警報の記録の変化点・区間の取込みに使う。帰属先は、現在のモデルのまま（現在のモデルが履歴の基準を持たなければ新しいモデル）、保持モデルの再利用、新しいモデル（候補の採用。[新規モデル作成](new-model-creation.md)）のどれか。

設定`training_data_assignment_settings.change_interval_search`の値は2つ。2026-10-29のユーザー判断により、基準版16から逆向きの1本の流れで1件刻みに測る扱いを既定にした（[spec](../../.kiro/specs/fedsda-every-sample-change-interval-search-default/design.md)）。

- `every_sample_with_reverse_prequential_code`（既定）: 列Wを新しい方から古い方へ並べ、学習者ごとに1本の流れで測る。その途中までの和を使い、最小変化区間以上・Wの長さ以下の全部のkを1件刻みで比べる。初期化し直した学習者のseedはclientの番号と警報の標本の番号だけから導き、kには依存しない。共有部の固定とrunの乱数の隔離はどちらの扱いでも同じ。Wが最小変化区間より短ければ学習者を使わず、候補は空で「現在のモデルのまま」を選ぶ（その後の件数不足の判定は同じ）。最小記述長の選択と組み合わせて使う。
- `minimum_interval_multiples_with_forward_prequential_code`（基準版15までの既定）: kは最小変化区間の倍数。kごとに末尾k件を古い方から、一から学習して測る。初期化seedは、clientの番号・警報の標本の番号・変化区間の長さkから導く。構成`fedsda-minimum-interval-multiples-change-interval-search`で選べる。

構成`fedsda-every-sample-change-interval-search`は、基準版16から既定の`fedsda`と同じ設定として、これまでのコマンドと結果の記録のために残す。2つの扱いの間で、現在のモデル・保持モデルの符号長、切り替わりの符号長、同じ長さのときの優先順は変えない。

warm startの学習者の寄与は、構成`fedsda-without-warm-started-learner`で比べられる（[spec](../../.kiro/specs/fedsda-without-warm-started-learner-configuration/design.md)）。最終提案の既定から`new_model_hypothesis_learners`だけを`reinitialized_learner`へ替え、「新しいモデル」の仮説を初期化し直した学習者だけで測る。学習者が1つなので、学習者の選択の符号長はln 1 = 0で、ln 2を足さない。変化区間の探索を含むほかの設定と既定は変えない。

選択ごとに、run記録の表`alarm_interval_assignment_selection.*`（警報の標本の番号、選んだ仮説、変化区間の長さ、行き先のモデルのID、符号長、警報にしたか、「新しいモデル」を選んだときに短かった学習者——基準版11から——）を残す。サーバのクラスタリングの閾値（γ=0.1）は、この選択とは別に残る。
