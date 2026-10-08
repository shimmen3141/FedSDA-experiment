# NEW-001: 候補検証sessionの保持への反映で、拒否の文言が一部の入力に合わない

- 発見日: 2026-10-08（Task 1の独立レビュー、Claude Haiku 5.5の任意の指摘）。対象: 新実装`src/federated_learning_experiments/runtime/held_candidate_validation_progress.py::apply_alarm_response_to_validation_session_holder`。発見spec: held-candidate-validation-progress（検証commit `733994b`）。
- 種別: 例外の文言の不正確（挙動は正しい）。状態: 未修正。

## 内容

保持が空でないときに、候補検証中でない応答を受けると`ValueError("response without an active validation requires an empty holder")`を出す。この条件には、sessionを持つ「候補検証を開始した応答」も当たる（保持中に別の候補検証を開始した応答を受けた場合）。その場合、応答はsessionを持っているので、文言の「without an active validation」は事実と合わない。

拒否そのもの（保持を変えずにValueError）は設計どおりで、testは例外の型と保持の不変で判定している。通常の経路（`handle_alarm_occurrence`）では、保持中なら応答は必ず候補検証中になるので、この拒否は起こらない。

## 扱い

文言だけの問題なので、発見時は変更しなかった（sourceを変えると基準環境の検証をやり直すことになるため）。このファイルを次に変更するspecで、「保持中に受けられるのは候補検証中の応答だけ」という意味の文言へ直す。修正commitは未定。
