# LEGACY-014: 採用時に新モデルへ移す保留標本が割当概念計数と損失統計へ反映されない

- 発見日: 2026-10-07。対象: 旧clients/fedsda.py::_finalize_forward_validationの採用分岐、clients/base.py::_absorb_into_store、servers/clustering.py::_oracle_concept_labels、固定基準748c3aa。
- 発見spec: adopted-candidate-local-adoption。種別: 分岐間の扱いの非対称。意図した仕様か未確認。状態: 再現済み・未修正（改善要否は未判断）。

## 再現と観測

実旧`SharedBackboneClassConditionalESRFedSDAClient`へ、真の概念1の保留標本4件（held_data）を持つ実ForwardValidationSessionを与え、採用条件で実_finalize_forward_validation(57)を呼ぶ。構築は tests/refactoring/test_adopted_candidate_local_adoption.py の build_local_adoption_oracle(pending_sample_count=4) と同じ。
2026-10-07基準venvで実行し、新モデル-103について次を観測した。

- train_data_store[-103]には保留標本4件が入る。
- model_concept_counts[-103]は空のまま（既存モデルは{4:{0:1}, 9:{0:1}}で不変）。
- model_stats[-103]の件数は初期統計batchの5件のままで、保留標本4件は加算されない。

採用分岐は`self.train_data_store[temp_id].extend(session.held_data)`だけを行う。同じメソッドの棄却・再利用・維持分岐と他の経路は_absorb_into_storeを使い、標本ごとに学習store追加・_record_model_concept・損失評価・_update_model_statsを行う。通常の標本処理（FIFOからの確定）も概念計数と統計を更新する。

## 影響

- 割当概念計数は実験診断専用で、servers/clustering.pyの_oracle_concept_labelsがモデルごとの多数概念の決定に使う。採用直後の新モデルは、後続の確定標本が入るまで計数が空で、oracleラベルが付かない（「未観測のモデルにはラベルを与えず統合しない」）。保留標本ぶんの計数が欠けることで、同数首位や多数概念の判定が変わる可能性がある。通常の予測・学習判断には使われない。oracle系の診断値・oracle_conceptクラスタリング判定への実際の影響は未確認。
- 損失統計: 初期統計は候補の学習区間（training_x/y）から作られる。保留標本を統計へ加えないことが、監視基準値・サーバ集約の件数加重へ与える影響は未確認。保留標本と学習区間の重なり（同じ標本を二重に数えないための意図的な扱いである可能性）も未確認。
- 過去実験成果への影響は未確認で、過去成果が誤っているとは判断しない。

## 今回の扱い

新実装のadopt_candidate_as_current_training_modelは旧と同じく、保留標本を学習標本storeへ追加するだけで、統計と割当概念計数を変更しない。実旧との一致をtestで固定している（既存モデルと新モデルの計数・統計の不変）。旧production・goldenは修正しない。

## 将来の検討

意図した仕様かを確認する。変更する場合は、採用分岐でも保留標本の真の概念を計数する（診断だけの変更）案と、統計も更新する（アルゴリズムの値が変わる）案を分けて判断する。後者はgoldenと過去実験の再検証が必要。担当/修正commitは未定。

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

損失統計の側は、問題なしとして閉じる。旧`_resolve_drift`は、候補の学習区間（`bx`・`by`）と保留標本（`held_data`）を、同じ`drift_data`から作る。初期統計は、その学習区間から作るので、保留標本は、すでに統計に入っている（加算すると二重計数）。計測（2026-10-13、新実装`ab10ad4`へ読取りだけの計数を差し込んだ。提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0と1の計8 run、Windows。8 run・1つの規模だけなので、頻度は目安）でも、採用48件の全部で、保留標本と初期統計の標本が同じだった。割当概念計数の側（診断）だけ、採用（3番目）: 採用のときに、保留標本の真の概念を数える。診断の値（oracleのラベル）だけが変わる。

## 時期の決定（2026-10-13、ユーザー）

割当概念計数の側（採用のとき、保留標本の真の概念を数える）は、診断だけの修正だが、概念計数から作る診断（oracleのラベルほか）は、重要動作のgolden（基準版1〜3）の記録の列に入っている。直すと、古い基準版の照合が通らなくなるので、いまは直さない。**対応しないのではなく、旧実装を外したあと（古い基準版を、旧と同じ扱いで照合し続ける必要が無くなった後）に直す候補**とする。損失統計の側は、上の「判断（2026-10-13）」のとおり、問題なしで閉じている。
