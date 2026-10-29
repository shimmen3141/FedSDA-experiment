# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を設計・評価し、論文（FedHASP、PAKDD 2027）に書く研究者（ユーザー）。
- 今の状況: 警報の後の最小記述長（MDL）の選択は、既定（基準版15）で、変化区間の長さkの候補を最小変化区間（既定5件）の倍数に限り、「新しいモデル」の仮説の予測的符号長を、kごとに末尾k件を古い方から一から学習して測る。
  逆向きの1本の流れで1件刻みに測る扱い（設定`change_interval_search=every_sample_with_reverse_prequential_code`、構成`fedsda-every-sample-change-interval-search`）は、前のspec（`fedsda-reverse-prequential-change-interval-search`）で選択肢として足した。
- 何を変えるか: 最終構成の既定を、逆向きの1本の流れで1件刻みに測る扱いにし、基準版16を足す。今の扱いは選択肢として残し、比べるための構成を足す。
- 背景: [IMPROVE-032](../../../docs/research/improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md)の本規模の結果（2026-10-29、uranium、10データ×4間隔×seed 0〜9）。
  既定との精度差は平均+0.01〜+0.02ppで、seedのばらつきの範囲。通信・MACの比は0.95〜1.05。
  ユーザーは、説明のしやすさ（kの候補に5の倍数という恣意的な刻みが要らない）、ロジックの自然さ（全部のkを1本の流れの途中までの和で測る）、計算量（測定用の学習者の更新が警報あたり列の長さ以下）を理由に、既定にすると決めた（2026-10-29「R1を既定にしたい」）。
  同じ日の「締め切りの前には既定を変えない」という判断は、この決定で置き換える。

## Introduction
設定の値と呼び方の対応:

| 設定 | 値 | この文書での呼び方 |
|---|---|---|
| `TrainingDataAssignmentSettings.change_interval_search`（既存の項目） | `minimum_interval_multiples_with_forward_prequential_code` | 「今の扱い」（基準版15までの既定） |
| 同上 | `every_sample_with_reverse_prequential_code` | 「新しい扱い」（基準版16からの既定） |

用語:
- **列W**: 警報の時点の帰属の保留（FIFO）の標本の列（警報の標本を含む、古い順、最大は容量＋1件）。
- **最小変化区間の倍数でないk**: 選択の記録で、選んだ仮説が保有モデルか新しいモデルで、変化区間の長さkが最小変化区間（既定5件）の倍数でないもの。今の扱いでは起きず、新しい扱いでだけ起きる。

## Boundary Context
- 対象: 最終構成の既定の値、構成の扱い（新しい扱いの構成の名前の扱い、今の扱いを選ぶ新しい構成）、最小記述長の選択を使わない設定の組（旧と比べる入口、実旧との照合の組、testの設定）での今の扱いの明示、基準版のgolden（版1〜15の照合を保つことと、版16の追加）、既定に依るtest、研究ツールの基点の固定と意味の保持、文書。
- 対象外: 2つの扱いの計算そのもの（前のspecのまま）、検出器、最小記述長の選択のほかの部分、件数不足の判定、帰属の保留、候補の学習と採用、登録、クロス評価、統合、予測、FedDrift・FedAvg、固定旧と旧golden、本規模の実験（必要ならユーザーが別に決める）、処理の流れのHTML（別のbranch）、論文の原稿。

## Requirements
### Requirement 1: 既定を新しい扱いにする
**Objective:** As a 研究者, I want 最終構成の既定が新しい扱いであること, so that 論文で説明する手法と既定の実装が同じになる。
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定の`change_interval_search`を「新しい扱い」にし、FedSDAの全部の構成（全部が最小記述長の選択を使う）が引き継ぐ。
2. The FedSDA system shall 構成`fedsda-every-sample-change-interval-search`を、既定と同じ設定の構成として残す（これまでのコマンドと結果の記録のため。`fedsda-synchronized-local-mixture-prediction`と同じ扱い）。runの識別子は構成の名前と完全な設定から作るので、この構成の識別子は前のspecのときと同じ（実行済みの結果、例えば`circle2_interval100_seed0_f07c86311252`と同じ）で、構成`fedsda`の識別子とは違う。
3. The FedSDA system shall 今の扱いを選ぶ構成`fedsda-minimum-interval-multiples-change-interval-search`を足す（最終構成の既定から、`change_interval_search`だけを「今の扱い」に替える）。goldenの事例にせず、専用のtestで守る。
4. The FedSDA system shall 全部の構成の組立てが例外を出さず、実験のコマンドの引数を変えない。
5. The FedSDA system shall 結果の重複の確認は今の仕組み（同じコード・golden・run設定）のままにする。基準版が16になるので、版15で実行した結果は、識別子が同じでも重複にならない（比べるときの対応はデータ・間隔・seedで取る）。

### Requirement 2: 最小記述長の選択を使わない設定の組で、今の扱いを明示する
**Objective:** As a 研究者, I want 既定を変えても、過去の照合と比較が同じ意味で動くこと, so that 既存の照合を失わない。
#### Acceptance Criteria
1. The FedSDA system shall 最小記述長の選択を使わない設定の組（`alarm_interval_assignment_policy`が最小記述長でないもの: 旧と比べる設定の入口、実旧との照合の組、testとfresh scriptの設定）で、`change_interval_search`を「今の扱い」と明示する（組合せの検査は前のspecのまま。新しい扱いと最小記述長でない選び方の組は拒否する）。
2. The validation shall 実旧との照合test（部品・全体run・構成）を、値を変えずに成功させる。

### Requirement 3: これまでの基準を保つ
#### Acceptance Criteria
1. The validation shall 基準版1〜15のgoldenを、「今の扱い」で、値を変えずに照合し続ける（`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`の今の扱いの明示は前のspecのまま。全部の版を、既存の版ごとの設定の組立てと照合のtestで確かめる）。
2. The validation shall 構成`fedsda-minimum-interval-multiples-change-interval-search`の設定が、基準版15の構成`fedsda`の設定と同じであることを確かめる（構成の名前とrunの識別子を除く）。

### Requirement 4: 新しい既定が、前のspecの構成と同じ値であることを確かめる
#### Acceptance Criteria
1. The validation shall 既定を変える前（このspecの基点のcommit）の構成`fedsda-every-sample-change-interval-search`の全体runと、変えた後の構成`fedsda`の全体runの、同じ条件の結果が、許容差なしで一致することを、同じ実行環境（Windowsの基準環境: `../../venv/Scripts/python.exe`、OMP/MKL/OPENBLAS各1 thread、同じMNISTの配置、`PYTHONPATH=src;.`、同じ機械で続けて実行）と、製品の掃引の入口（`run_experiment_sweep.py`、`--workers 1`、10 client）で、複数のデータ（MNIST4、SINE2、SEA4、CIRCLE2、間隔50、seed 0、各client 1000標本と、列Wが最小変化区間より短い警報が起きるSINE2・seed 1）で確かめる。一致の定義:
   (a) 記録の表の全配列が、同じ名前の配列と`numpy.array_equal`（浮動小数は`equal_nan=True`）で一致する。
   (b) runの要約の全項目が一致する。除くのは、実行の時刻・秒数・出力先の名前・コードの来歴・基準版の番号・構成の名前・runの識別子。
   (c) 最終のパラメータ（rawに保存される全部のモデルの全部のパラメータ）が一致する。
   比べる記録は、製品が1 runごとに保存するrawの要約（JSON）と記録の表・パラメータ（NPZ）の全部とする。データの生成・乱数はrunの設定（seed）だけから決まるので、条件の一致は設定の一致で確かめる（(b)の除外を除いた設定の全項目が同じ）。

### Requirement 5: 新しい既定を基準版16として足す
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版16のgoldenとして、Windowsの基準環境とWSLの両方で足す（ケースは版15と同じ18ケース）。各環境で2回の独立したprocessの一致と必須の経路を確かめて生成する。
2. The validation shall 版15と16の差を、goldenを使わず両方の版をその場で実行して確かめる: FedDrift・FedAvgのケースは、指標・最終のパラメータ・記録の表・経路の記録（版16で足す項目を取り除いたもの）が版15と同じ。FedSDAのケースは、設定が`change_interval_search`だけ違う。
3. The validation shall 経路の記録に、最小変化区間の倍数でないkの選択の回数を足し（版16から。FedSDAのケースだけ）、新しい既定を引き継ぐFedSDAのケースのうち2つで1以上であることを、必須の経路として確かめる。2つのケースの名前は、実装の前（task 1）に、版15の18ケースのうちFedSDAのケースの設定の`change_interval_search`だけを新しい扱いに替えてWindowsの基準環境で実行し、次の規則で決めて記録する: 第一候補の`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`のうち回数が1以上のものを採り、足りない分は、版15のケースの並び（`behavior_golden_cases.py`の版15の一覧の順）で、回数が1以上の最初のケースから補う。2つに満たなければ、ケースの条件を変えずに、ユーザーへ報告して扱いを決める。WSLでの生成の前にも、同じ2ケースの回数が1以上であることを確かめる。版15までの必須の経路は、版16でも求める。

### Requirement 6: 研究ツールの意味を保つ
#### Acceptance Criteria
1. The research tool shall 変種を次の組に分け、組ごとに`change_interval_search`の値を決める（`tools/research_provisional_candidate_expert/run.py`の`_execute_job`）:
   - 組A（基準版10の既定を基点にする変種。`_PRODUCT_DEFAULT_BASED_VARIANTS`に無い変種。研究用CLIの`default`を含む）: 基点の固定に、今の扱いを足す。
   - 組B（基準版12へ戻すIMPROVE-031の変種。`LOSS_MONITOR_REBASELINE_VARIANTS`、`product-default-all-evaluating-clients`、`product-default-all-evaluating-clients-candidate-shared-feature-update`）: 基準版12への戻しに、今の扱いを足す。
   - 組C（IMPROVE-032の研究用の変種`default-reverse-prequential-grid5`・`default-reverse-prequential-grid1`・`default-reverse-prequential-grid1-position-cost`）: 今の扱いへ戻す。選択の関数の差し替え（kの候補・測り方・位置の項）は今のまま。列が最小変化区間より短い警報で製品の選択へ戻すときも、戻した今の扱いを使う。
   - 組D（`product-default`と`every-sample-change-interval-search-configuration`）: 戻さない（製品の既定・構成のまま。基準版16では両方とも新しい扱い）。
   - 組E（ほかの`product-default`系の変種）: 戻さない（版13のspecと同じく、製品の既定、基準版16の上で評価する。意味が変わることをコメントに書く）。
2. The research tool shall 測定用の学習者の更新回数の測定（`reverse_prequential_policy`、`prequential_measurement`）を、runの`change_interval_search`が今の扱いのときと組Cだけで行う。新しい扱いの組D・組Eでは測定を差し込まず、`prequential_measurement`は空にする（0回の意味ではない。前のspecの`every-sample-change-interval-search-configuration`と同じ扱い）。
3. The validation shall 研究ツールの全部の変種（`_VARIANTS`の全部の名前）について、既定を変える前と後で、`_execute_job`が製品の全体runへ渡す設定（製品の全体runの入口を包んで設定を取り出し、runは実行しない）と、差し込む研究用の差し替え（`reverse_prequential_policy`ほかのcontextを入れるかどうか）を取り出して比べる:
   - 組A〜C: 設定の全項目と差し替えの有無が、前と後で同じ。
   - 組D・E: 設定の差が`change_interval_search`の1か所だけ（前は今の扱い、後は新しい扱い。`every-sample-change-interval-search-configuration`は前後とも新しい扱いで差なし）で、差し替えの有無の差は`reverse_prequential_policy`だけ（後は入れない）。
   - どの組にも入らない変種、2つ以上の組に入る変種が無い（組は`run.py`の一覧から機械的に決める）。
   - 研究用の入口`tools/research_loss_monitor_rebaseline.py`の全部の変種（その`_VARIANTS`。`product-default`は組D、IMPROVE-031の変種は組B）も、同じ方法で取り出して同じ規則で比べる。
   前の取り出しはtask 1で保存し（Git管理外）、後の取り出しと比べる一時scriptの結果を記録する。
   この比較が保証するのは、runへ渡す設定と、差し込む研究用の差し替えが変わらないこと（組D・Eは決めた差だけ）までである。差し込んだ後の研究用の差し替えのうち、`change_interval_search`を読むのは`reverse_prequential_policy`（短い列で製品の選択へ渡す引数）だけで、ほかは読まない（研究ツールのsourceで`change_interval_search`を読む箇所が、検査のtestを除いてこの1か所だけであることを、task 2の後に検索で確かめて記録する）。製品の全体runは設定と乱数のseedだけで決まるので、設定と差し替えが同じなら結果も同じになる。runの結果まで同じことは、下の4で代表の変種について確かめる。研究ツールの検査には、組の決め方と代表の変種の値のtestを足す。
4. The validation shall 既定を変える前と後の、研究ツールの小さい条件（SINE2・seed 0・10 client・600標本・間隔100と、MNIST4・seed 0・10 client・600標本・間隔100の2条件）の結果を比べる（秒数・時刻・コードの来歴を除く全部の出力）:
   上の3で設定と差し替えが全部の変種で守られることを確かめ、ここでは代表の変種で、runの結果まで同じことを確かめる。
   - 同じであること: 研究用CLIの`default`、組Cの3変種、`product-default-loss-monitor-decrease-rebaseline`、`every-sample-change-interval-search-configuration`、研究用の入口（`tools/research_loss_monitor_rebaseline.py`）の`product-default-all-evaluating-clients-loss-monitor-decrease-rebaseline`。
   - 決めた差だけ替わること: 組Eの代表`product-default-class-conditioned-prediction`と、研究用の入口の`product-default`は、後の結果が、前に同じ変種へ`change_interval_search`＝新しい扱いだけを当てた結果（task 1で、一時scriptで設定のその1か所だけを替えて実行して保存する）と同じ。
   - 製品の既定が替わったこと: 後の`product-default`が、前の`every-sample-change-interval-search-configuration`と同じ（測定の`prequential_measurement`はどちらも空）で、前の`product-default`と違う（変化区間の長さが最小変化区間の倍数でない選択が1回以上ある）。

### Requirement 7: 文書を合わせる
#### Acceptance Criteria
1. The documentation shall 手法の説明（提案手法の概要、モデル再利用の文書の最小記述長の節）、基準の文書（版16）、実験の入口（構成の一覧）、IMPROVE-032の状態、再開案内で、新しい既定を書く。過去の日付の記録は書き換えない。
