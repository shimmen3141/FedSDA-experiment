# Implementation Plan

- [x] 1. DWMの規則を純粋な関数として提供する
  - 投票（同点は最小のクラス）と、周期の標本での重みの更新（誤った専門家の減衰→投票し直し→最大の重みでの正規化→θ未満の削除→追加の判断）を、学習と乱数から切り離して実装する。不正な入力（専門家0、長さの不一致、範囲外のクラス、非有限の重み）を拒否する。
  - 追加の判断に使う投票は、減衰の直後・正規化と削除の前の重みで行う（削除の後に投票し直すと結果が変わる例を手計算の例に入れる）。
  - 完了条件: `tests/refactoring/test_dynamic_weighted_majority_rules.py`が、手で計算した例（全員正解で重みが変わらない、全員誤りで正規化により最大が1に戻る、θ未満の削除、投票し直した予測が誤りのときだけ追加、同点）と拒否の例で通る。
  - _Boundary: methods/dynamic_weighted_majority（規則）_
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2. DWMのclientを実装する
  - 専門家（ID、分類器、optimizerの状態、保持標本、重み、学習の計数）・記録（標本ごとの予測、専門家の作られた・消された位置）を所有し、design「System Flows」の1〜5の順で標本を処理する。新しい専門家は事前学習の直後の状態の複製（分類器とoptimizerの組を1回の`deepcopy`で。design参照）。完了条件に、足した専門家が実際に学習される（パラメータが変わり、元の初期状態は変わらない）ことを含める。学習は`update_independent_model_parameters`と`python_random_generator.sample`をFedAvgのclientと同じ順で呼ぶ。実行の枠の操作（区間末の消化ほか）は何もしない。
  - 完了条件: `tests/refactoring/test_dynamic_weighted_majority_client.py`で、(a) p>標本数のとき同じ初期状態・同じseedの`FedavgClient`と予測の列・全パラメータ・optimizerの状態・Pythonの乱数の状態が一致、(b) 周期の標本でだけ重みと専門家が変わり記録と一致、足した専門家の保持が作られた標本から始まり初期状態の複製で始まる、(c) 評価の予測が学習の前の重みによる、(d) 真のコンセプトを替えても予測・学習が変わらない、(e) 不正な入力で状態と乱数が変わらない、が通る。
  - _Boundary: methods/dynamic_weighted_majority（client）_
  - _Depends: 1_
  - _Requirements: 1.1, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 3.1_

- [x] 3. DWMのrunの設定・サーバ・参加者の準備を実装する
  - 参加者の設定とrunの設定（β・θ・pの範囲の検査を含む）、`fedavg`の既定から値を取り出してβ=0.5・θ=0.01・p=50を足す既定の組立て、同期で何もせず通信量0を記録するサーバ、事前学習1回でclientとサーバを準備するfactoryを作る。
  - 完了条件: 既定の組立てが`fedavg`と同じ実行・事前学習・構造・optimizer・batch・更新の回数・指標の設定を持つことと、範囲外のβ・θ・pの拒否、factoryが各clientへ同じ初期状態（事前学習の直後）を渡すこと、サーバの同期で通信量が0のままであることをtestで確かめる。
  - _Boundary: runtime（DWMの設定・サーバ・参加者）_
  - _Depends: 2_
  - _Requirements: 1.1, 1.2, 2.5, 3.1, 3.2, 3.3_

- [x] 4. DWMの全体run・指標・記録の表・保存を実装する
  - 計測つきの全体run、FedAvgと同じ項目の指標の導出（design「runtime」の各項目のDWMでの意味、検出0件の検出の指標）、design「Data Models」の表の対応どおりの記録の名前付きの配列、JSON・NPZの保存と読込み（`method_name`の確認を含む）を作る。
  - 完了条件: `tests/refactoring/test_dynamic_weighted_majority_experiment_run.py`で、SINE2・SEA2の小さい条件の全体runの指標が記録から導いた値と一致し、通信量0、保存→読込みの往復で値が一致し、既存ファイルがあるときに上書きせず拒否する。
  - _Boundary: runtime（DWMの全体run・記録・保存）_
  - _Depends: 3_
  - _Requirements: 3.2, 4.1, 4.2, 4.3, 4.4_

- [x] 5. 構成`dwm-neural-network`を掃引へ接続する
  - 構成名の定数を一覧の末尾へ足し、設定の組立て・計画の型・実行と保存・既存の結果の読込み（FedSDAの読込みへ落ちる前の明示の分岐）へDWMを振り分ける。構成の名前の一覧を固定で照合している既存のtest（`test_fedsda_configuration_variants.py`の完全一致と変種の集合の差、`test_fedsda_configuration_candidate_settings.py`の設定の型の除外、`test_soft_training_single_model_prediction.py`の末尾の照合）と、`behavior_golden_cases.py`の`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`へ、design「Modified Files」のとおりDWMを足す（既存の名前・順・goldenは変えない）。`docs/experiments/experiment-sweep.md`の構成の一覧へ、目的と`fedavg`・論文のDWMとの違いを書く。
  - 完了条件: `tests/refactoring/test_dynamic_weighted_majority_experiment_sweep.py`で、計画・実行・保存・一覧のCSV・再開（完了済みの飛ばし、片方だけのファイルの競合）が通り、FedDrift系の軸を指定してもDWMのrunが増えず、既存の構成の名前・順・runの識別子が変わらない。上の既存の4ファイルのtestと`test_important_behavior_regression_goldens.py`が通る。DWMの分岐が設定の組立ての既定の経路（FedSDA）より前にあり、保存の振り分けでどの型にも当たらなければ例外になる。`test_dynamic_weighted_majority_experiment_run.py`が、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`で`dwm-neural-network`が自分のファイルに割り当てられていることを確かめる（既存の保護対象の専用testと同じ形）。
  - _Depends: 4_
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. 依存の境界・fresh process・全体の検証
  - 新しいmoduleの依存の許可集合を`tests/refactoring/test_single_run_dependency_boundaries.py`へ登録し、`tests/refactoring/fresh_process_smoke.py`へDWMの小さい全体runと保存・読込みの流れを足す。naming.mdを実装に合わせ、`spec_checks.py names`を通す。
  - 完了条件: 依存の境界のtestとfresh process smokeが通り、全pytest（goldenを含む）・Ruff・pyrightが通る。既存のgoldenとrunの識別子は変わらない。件数とcommitをImplementation Notesへ記録する。
  - _Depends: 5_
  - _Requirements: 5.3_

## Implementation Notes

### 仕様書群の独立レビュー（1回目、2026-11-17）
- 担当: GPT-6 Lunaで開始したが判定が得られず（spec.jsonの`specification_review.fallback_reason`）、Claude Haiku 5.5（effort medium、読むだけ、session `13dfb876-a8df-4a65-875d-d89ed3a765ad`、`is_error=false`、`modelUsage`はclaude-haiku-5-5）へ代替。判定FAIL（Major 4、Minor 8、任意1）。
- 採用（Major）: (1) 切替の検出の指標は、FedAvgと同じく検出0件として共通の関数で計算する（要求4.1を直した）。(2) 保存する表を、FedAvgの表との対応表としてdesign「Data Models」に書いた（`held_model_count_at_prediction`・`prediction_model_id`の名前をFedAvgに合わせ、`model_loss_statistics`は無し、`expert_lifetime`を足す）。(3) 専門家の初期のoptimizerは`concept_specific_parameter_optimizer_state`（FedAvgと同じ）で、clientの中で元を複製して保つことを書いた（要求1.2も具体化）。(4) 構成名の一覧を照合している既存のtest 3ファイルと`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`をModified Filesとtask 5に特定した。
- 採用（Minor）: (5) `updates_per_sample`は`fedavg`の値を使い、局所更新をまとめる間隔の設定は持たず毎標本に固定（要求1.1・design）。(6) 図1のiは`sample_index + 1`、記録の位置は0からの`sample_index`。(7) サーバの3つの操作を書いた。(8) 計算の計数の範囲（損失の統計のforwardが無い）と、計測の包みをDWMのmoduleに非公開で書くことを決めた。(9) `final_global_model_count`は共通の列の名前として残し、DWMでの意味を書く。(10) `DynamicWeightedMajorityExpert`をdesignに定義し、残った専門家の消された位置を-1にそろえた。(11) 参照の場所を元のcheckoutの絶対パスで書いた。(12) `method_name`を`dynamic_weighted_majority`に決め、既存の結果の読込みをFedSDAへ落ちる前の明示の分岐にした。
- 任意（13）: 区別は十分という確認。変更なし。
- 実装・結果に影響するMajorを直したので、同じ段階の担当（Haiku）に修正箇所の再確認を依頼する。

### 仕様書群の再確認（2回目、2026-11-17）
- 担当: Claude Haiku 5.5（effort medium、読むだけ、session `2267f727-3b91-4e66-8438-d12aeff8a5a1`、`is_error=false`、`modelUsage`はclaude-haiku-5-5）。判定FAIL（Major 2、Minor 2、任意1）。前回のMajor (1)〜(3)は解消、(4)は1ファイル漏れと確認された。
- 採用（Major）: (1) 全構成名を回す`test_fedsda_configuration_candidate_settings.py`の除外への追加をModified Filesとtask 5に足した。(2) 専門家の複製を、FedAvgと同じく分類器とoptimizerの組の1回の`deepcopy`にすると明記し、task 2の完了条件に、足した専門家が実際に学習されることを足した。
- 採用（Minor）: (3) 通信量・サーバの計数・統合の計数・計算量の要約のDWMでの値を書いた。(4) 学習の計数は専門家が持ち`model_training`の表へ書く（寿命の記録からは外した）。
- 採用（任意）: (5) 全構成名を回す既存のtestが変更なしで通ることを、掃引のtestの観点に足した。
- Majorを直したので、再び修正箇所の再確認を依頼する。

### 仕様書群の再確認（3回目、2026-11-17）
- 担当: Claude Haiku 5.5（effort medium、読むだけ、session `99adb187-07b9-49ff-9bca-05135164bdf2`、`is_error=false`、`modelUsage`はclaude-haiku-5-5）。判定PASS（Blocker・Majorなし、Minor 4）。
- 採用（Minor、文書だけの修正なので再確認なし）: (1) 全構成名を回す既存のtestの記述を「型で絞るものは通る、型で絞らないのは1件だけ」に正確にした。(2) DWMの分岐を設定の組立ての既定の経路より前に置き、保存の振り分けで型に当たらなければ例外にすることを書いた。(3) 追加の判断の投票が減衰の直後・正規化と削除の前であることを要求2.2・design・task 1に書いた。(4) 専用testで保護の表への所属を確かめることをtask 5に書いた。

### 実装（2026-11-17）
- commit: task 1・2 `84a35445`、task 3・4 `1259513d`、task 5 `4dce41bd`、task 6の登録 `06cc2ca2`。main の fixture の修正（下）を `bb2310dd` で取り込んだ。
- 手順の逸脱: task 3は、testより先に実装を書いた（testは後でtask 4と同じファイル`test_dynamic_weighted_majority_experiment_run.py`にまとめて書いた）。
- 新しいtest: 規則23、client 13、全体run 13、掃引4（うち1件はslow）。依存の境界は、新しいmoduleの許可集合（末端の名前だけ）とexact検査の一覧への追加、掃引の3 moduleの許可集合へのDWMの名前の追加。fresh process smokeにDWMの全体run・保存・読込みを足した。

### 実装レビュー（2026-11-17）
- 担当: Claude Haiku 5.5（effort medium、session `07d4e460-b725-4a47-97e8-a2323ac06c18`、`is_error=false`、`modelUsage`はclaude-haiku-5-5）。GPT-6 Lunaは使用量の上限（22:00ごろまで回復しない）で使えず、実装のレビューの優先の担当でもあるHaikuにした。判定PASS（DWMの差分に起因するBlocker・Majorなし）。レビュー担当は新しい4ファイル（53 passed）、依存の境界・構成の照合の既存test（2406 passed、1 failed＝下の既存の失敗）、Ruffを独立に実行した。全pytest・fresh smoke・pyrightは独立に実行していない。
- 指摘1（Major、差分の外・既存）: 保護するgoldenのhashのfixture（`soft_training_single_model_prediction_existing_configurations.json`、2026-11-07）が、CRLFの内容のhashを記録していた。ユーザーの承認（2026-11-17）を得て、mainで2件（`feddrift_regression_goldens/linux-…json`、`golden/fedsda_without_candidate_validation_random_states.json`）をLFの内容のhashへ直した（main `58c5d571`、このブランチへ`bb2310dd`で取り込み）。goldenの中身は変えていない。
- 指摘2（Minor）: 保存の振り分けの`else: raise TypeError`は`try`の中なので、runの失敗として記録される。手前の型の検査で先に止まるので到達しない。変更しない。
- 指摘3（任意）: 専門家ごとに全保持標本を持つので、メモリは専門家の数×標本の数で増える（要求1.4のとおり）。大きいrunでの注意として記録する。

### 全pytest（2026-11-17、HEAD `bb2310dd`、Windows、12 worker、`--dist loadfile`）
- 13,847 passed、8 skipped、1 failed（15分16秒）。JUnit: `venv/refactoring-tests/dwm-neural-network-baseline-full.xml`。
- 失敗の1件は`test_research_method_pareto_figure.py::test_statistics_and_outputs`で、Windowsのアプリケーション制御がmatplotlibの使うfontToolsのDLL（`iup`）の読み込みを止めた（`ImportError: DLL load failed ... アプリケーション制御ポリシーによってこのファイルがブロックされました`）。コードの不具合ではなく、単独で再実行して通過した。全pytestを通しで再実行はしていない。
- `spec_checks.py identity`: `tests.test_regression`・`tests.test_proposed_regression`は成功、作業ツリーに未コミットの差分なし。「固定旧基準748c3aaからの差分が空」はNGだが、差分は過去のspec（2026-11-03〜11-07）が入れた旧実装の`experiment_spec`の3ファイルと、`tools/`の研究ツールで、このspecの前（`b9368285`）から同じ。このspecは旧実装と旧のgoldenに触れていない。
- Ruff（変更したsourceとtest）、pyright（変更したsource、0件）、fresh process smokeは成功。

