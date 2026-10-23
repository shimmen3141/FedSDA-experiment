# 研究の引継ぎ（2026-10-23時点）: 統合の判定・共有の研究と、基準版13の候補

この文書は、別のAIへ引き継ぐとき、またはトークンが回復して再開するときの入口。2026-10-22〜23の研究（IMPROVE-028を中心とする、サーバの統合と、モデルの間の共有の研究）の現在地・前提・分かったこと・残る判断・次の一手をまとめる。詳細な数値・経緯の正本は[IMPROVE-028](improvement-candidates/improve-028-evidence-accumulating-model-consolidation.md)（1〜17節）、実験の一覧は[実験の台帳](../experiments/experiment-log.md)、リファクタリング全体の現在地は[再開案内](../../.kiro/steering/resume.md)。

## 1. 前提

### 作業場所と状態
- worktree `C:\Users\yshin\Research\FedSDA-experiment\.worktrees\refactoring`、ブランチ`refactor/architecture`（mainへ未マージ）。2026-10-23の終わりのHEADは`95fb0dcd`以降で、作業ツリーに未commitの変更は無い（開始時に`git status`で確かめる）。
- 動作基準は**基準版12**（[spec](../../.kiro/specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)、完了）。FedSDAの既定は、基準版11（「新しいモデル」の仮説の2つの学習者、初期モデルの統計の5分割交差検証）に、クロス評価で対象の標本を返せるclientだけを評価させる扱い（`cross_evaluation_client_selection_policy=clients_with_target_model_samples`。LEGACY-022の修正）を足したもの。構成`fedsda-with-candidate-validation`は、基準版10までの扱いを明示する。

### 研究ツール
- `tools/research_provisional_candidate_expert/run.py`。変種は`_VARIANTS`に登録。**`_PRODUCT_DEFAULT_BASED_VARIANTS`に入れた変種（`product-default`で始まるもの）は、製品の既定（基準版12）の上で動く。それ以外の変種は、基準版10の既定へ戻して動く**（`_execute_job`の基点の固定。基準版11・12で足した項目を、版10の値に置き換える）。新しい変種を製品の既定の上で試すときは、`product-default-…`の名前にして、その組へ足す。
- 研究ツールの結果は`results/fedsda-provisional-candidate-expert/<日付>_<名前>/results.json`（runごとの要約。記録の表——raw——は保存しない）。rawが要る診断は、一時scriptで`federated_learning_experiments.runtime.fedsda_run_record_tables.build_fedsda_run_record_tables`を包んで保存する（例: 会話の中で使った`run_with_model_losses.py`。Git管理外）。
- 研究ツールの`product-default-sample-holding-cross-evaluation`の280 run（`20261023_uranium-v11-sample-holding-cross-evaluation-*`）は、基準版12の製品の既定と全run一致することを確かめてあるので、**基準版12の既定の比べる相手として使える**。

### 実行環境とやり方（ユーザーとの取り決め）
- 長い実験は研究室サーバ（uranium。ユーザーは「研究室サーバ」とも呼ぶ）で、ユーザーが実行する。コマンドを渡す前に、`refactor/architecture`をpushしておく（通常のpushは確認不要）。コマンドの形: `tmux new -s …` → `source .venv/bin/activate` → `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` → コマンド（複数行は`\`）。workersは合計64程度。人へ渡すコマンドに`--print-plan`を入れない。
- 手元（Windows、16コア）の実験は、14 workers以内、2時間程度まで。MNISTを使う実行は`FDE_MNIST_DATA_DIR=C:/Users/yshin/Research/FedSDA-experiment/data/mnist`、`PYTHONPATH="src;."`。1時間以上かかる実験は、開始から57分後に進み具合を確かめる（1回目の確認で残りが1時間以内なら、2回目以降は不要）。
- **実装の分担（2026-10-23のユーザー指示、Claudeのトークン節約のため）**: 実装はGPT-6.1-sol（`codex exec -m gpt-6.1-sol --sandbox workspace-write "<依頼文>"`）に任せ、Claudeが差分を確認して、不備は直す。レビューはGPT-6 Luna（`codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox workspace-write --ephemeral "<依頼文>"`）。**`gpt-6.1-luna`は、このアカウントのCodexでは使えない**（"not supported when using Codex with a ChatGPT account"）。**Solのsandboxの中からは、codexの起動（アクセス拒否）とpytestの一時ディレクトリが使えない**ので、Lunaのレビューと全pytestはClaude（主担当）が実行する。依頼文はファイルに書き（`results/_sol_prompts/`、gitignore）、`"$(cat …)"`で渡す。Git Bashのheredocは、引用符を含む本文で壊れやすいので、scriptはWriteで書いてから実行する。
- 研究の記録: 不具合・改善候補は`docs/research/implementation-findings/`・`docs/research/improvement-candidates/`へ1件1ファイル。実験は[実験の台帳](../experiments/experiment-log.md)へ1行ずつ。

## 2. ここまでで分かったこと（要点）

1. **統合は、ほぼ「新しいモデルを登録した同期の、1回だけの判定」でしか起きていなかった**（IMPROVE-028の1〜3）。理由は2つ: (a) クロス評価で、標本を持たないclientを評価に選ぶ（LEGACY-022。基準版12で修正済み）、(b) クラスタリングを、新しいモデルの登録がある同期でしか行わない（旧の既定`on_new_model`を移植。`every_round`は移植していない）。
2. **SEA4の精度の低さは、概念2（閾値7.0）が専用のモデルを持てないrunがあること**（NEW-006の「概念2がどこで取り込まれるか」）。clientの側で概念2を分ける機会が少なく（区間の約2割）、分けても、登録の直後に概念0のモデルへ統合される。概念2は、clientの利用の実績でも概念0とほとんど区別できない（1標本あたり0.006〜0.03 nats）。統合の判定を変える方向では改善しにくい。
3. **今の統合の判定（クラス別の正誤の信頼下限≦0.1）は、MNIST・SINE2ではよく働く**。弱いのはSEA4（機能の上で区別しにくい概念）と、MNIST4のまれな崩れ（LEGACY-021）。基準版11で、MNIST4の崩れは大きく減った（88%未満の概念のrun 9→3、20 seed）。
4. **clientの利用の実績（予測の前の対数損失の、モデルの対ごとの累積差）は、計算の追加なしに大量の証拠を持つ**が、「若いモデルは同じ概念でも成熟したモデルの標本で劣る」「若いモデルがまだ学べていない間は、違う概念の成熟したモデルの方が良く見える」という学習の進み具合の混同があり、同じ概念かの判定・借りる価値の重みのどちらにも使いにくい（5・6・14・16）。
5. **パラメータを混ぜる共有（案B1）は、非線形性と上の混同のために、違う概念を壊す**。他のモデルの予測を教師にする蒸留（案C）も、標本ごとの判定では違う概念の教師を除けない。
6. **クロス評価は、基準版8から配布済みのモデルで行うので、評価させるclientを増やしても通信はほぼ増えない**（増えるのは軽いメッセージと、全体の計算の0〜2%）。

## 3. 試したことと結果・判断（IMPROVE-028の節）

| 試したこと | 結果 | 判断 |
|---|---|---|
| 新しいモデルの統合を遅らせる（1） | SEA4は上がるが、MNIST2・CIRCLE2で下がり、SINE2の通信2倍超 | 採らない |
| 評価のclientを、標本を返せるclientに限る（2・9） | 精度は中立、CIRCLE2の下り0.67〜0.81倍 | **基準版12として採用** |
| 上と、毎回の同期でのクラスタリング（4） | MNIST2の重複は統合されるが、SEA4の改善は消え、違う概念の統合が増える | 採らない |
| 案A①: 利用の実績での同じ概念の判定の見積り（5・6） | どの単純な規則も今の判定を上回らない | 判定の置き換えとしては採らない |
| 案A: 利用の実績による統合の拒否権（8・10・11） | 安全だが、基準版11の上では得が小さい（+0.0〜+0.16pp） | **既定にしない**（ユーザー判断。案B1で同じ証拠を使うなら一緒に考える） |
| 評価させるclientの上限を3→clientの数（13・14） | 精度−0.09〜+0.17pp、計算+0〜1.7%、通信ほぼ同じ | **基準版13の候補**（ユーザー: 外すのがよさそう、ただし今は版を上げない） |
| 案B1: 概念ごとの部分の、利用の証拠による方向付きの連続的な共有（12〜14） | 本規模で、SEA4・MNIST4で下がる（最悪−2.17pp）。借りた量の6〜7割が違う概念から | 今の形では採らない |
| 案C: 他のモデルの予測を、標本ごとに選んだ弱い教師にする蒸留（15・16） | CIRCLE2 −0.23pp、MNIST2 −0.13pp、MNIST4 −0.20〜−0.31pp。教師の5〜8割が違う概念から | 採らない |
| 候補の共有部の固定を外す（17、基準版12の上のablation） | CIRCLE2は4間隔とも+0.08〜+0.21pp、ほかは−0.17〜+0.19ppでばらつく（MNIST4・500で−0.14） | **基準版13に入れるかは、ユーザーの判断待ち** |
| 基準版13の候補（上の2つを合わせた形、17） | 精度−0.13〜+0.23pp、下り0.97〜1.02倍。2つの効果はほぼ足し算 | 同上 |

ほかの関連する記録: 先行研究の調査（deep research、ユーザー作成）は元のcheckoutの`docs/research/FedSDA案B1_B2の先行研究調査と新規性リスク評価.md`（**Git管理外**。要点はIMPROVE-028の12）。基準版11のablationは[こちら](fedsda-component-ablation-v11-20261023.md)。論文の主軸（「帰属・学習は硬く、予測は柔らかく」を明示的に分ける）は[method-positioning.md](method-positioning.md)の1。

## 4. 残っている判断（ユーザーが決める）

1. **基準版13に入れるもの**: (a) 評価させるclientの上限を外す（ユーザーは「外すのがよさそう」と述べた。損はほぼ無い）。(b) 候補の共有部の固定を外す（CIRCLE2は一貫して良く、ほかは中立〜わずかに悪い。精度の改善ではなく、規則を減らす単純化）。ユーザーは、トークン残量のため、版を上げる重いtaskを当面避けると述べた。
2. **次の研究の方向**: 案E（サーバでは共有せず、clientの予測の混合を強くする。主軸に最も沿う。具体案は[IMPROVE-029](improvement-candidates/improve-029-stronger-prediction-mixture-for-recurring-and-partial-concepts.md)。最初に試すのはE1——Fixed-Shareの配り直しの先を、一様から過去の重みの混合にする、Mixing Past Posteriors）か、SEA4の概念2をclientの側で分ける問題（具体案は[IMPROVE-030](improvement-candidates/improve-030-split-partially-changed-concept-on-client.md)。最初に、警報の後の選択で「新しいモデル」が僅差で負けているかを診断し、仮説どおりならS1——判定を保留して、影の学習者で予測的な符号長の累積差を積み、ln 2を超えたら採用——を試す）か。どちらも、文書に、前提・仕組み・差し込む場所・製品の既定と一致させる確かめ方・比べる相手の結果のディレクトリ・評価の指標を書いてある。
3. 以前から保留のもの: 候補検証の処理を消すか、Git管理外の3文書（AGENTS.mdの「文書とコミット保留」）の扱い、mainへの統合の時期、IMPROVE-025（損失の距離のクラスタリング。採らない方向）・IMPROVE-026（符号長の統合の判定。保留）・LEGACY-021（違う概念の統合。基準版11で頻度が減った）。

## 5. 残っているタスクと、次にやること（推奨の順）

1. **ユーザーに4の1・2の判断を仰ぐ**（判断が出るまで、版を上げるspecは始めない）。
2. **基準版13のspec**（4の1が決まったら）。手本は基準版12のspec（[fedsda-cross-evaluation-clients-with-target-samples](../../.kiro/specs/fedsda-cross-evaluation-clients-with-target-samples/)）。進め方は[共通引継ぎ手順](../../.kiro/steering/agent-handoff.md)（仕様書群→独立CLIの仕様レビュー→実装→goldenの生成（Windows・WSL）→独立CLIの実装レビュー→全pytest・Pyright・WSL・fresh smoke）。実装はSol、レビューはLuna、全pytest・goldenの生成はClaude（1の分担）。設計の要点:
   - 評価させるclientの上限: 最終構成の定数`_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL = 3`（`runtime/fedsda_final_configuration_run_settings.py`）を、上限なし（clientの数）にする。研究ツールの変種`product-default-all-evaluating-clients`は、参加者の設定`maximum_evaluating_client_count_per_model`をclientの数に置き換えている。製品で「上限なし」をどう表すか（値をclientの数にする／Noneを許す）は設計の判断。
   - 候補の共有部の固定: 最終構成の既定`candidate_shared_feature_update_policy`を`update_shared_features_with_candidate`にする（構成`fedsda-without-candidate-validation`と同じ値になるので、その構成の扱いを決める）。
   - 研究ツールの基点の固定（`_execute_job`）に、変えた項目の版12までの値を足す。研究ツールとの照合は、`product-default-all-evaluating-clients-candidate-shared-feature-update`（13の候補）と、製品の新しい既定が一致することで確かめる。
   - Pyrightは`../../venv/Scripts/pyright.exe --pythonpath "C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe"`で実行する（指定しないと依存を解決できず198 errors）。全pytestは`tests`を明示する（`results/`の下の`test-results.txt`がdoctestとして集められて失敗するため）。
3. **案E（IMPROVE-029）か、SEA4の概念2（IMPROVE-030）**（4の2で決まった方）: 各文書の「試し方（引継ぎ先へ）」の手順で進める。研究ツールの変種は`product-default-…`の名前にして`_PRODUCT_DEFAULT_BASED_VARIANTS`へ足し、実装はSol、確認はClaude、レビューはLuna。
4. 基準版12の既定の製品の入口での本規模（記録の表つき）は、2026-10-24に`results/method-comparison-v12/fedsda`（240 run、seed 0〜9）として取れた。研究ツールの結果と全run一致。rawで統合や帰属を調べるときは、これを使う（MNIST4のseed 10〜19のrawは無い）。

## 6. 主な結果のディレクトリ（`results/`、Git管理外）

- 基準版11の既定: `method-comparison-v11/20261022_221105_fedsda-v11-default-seeds0-9`、`…/20261022_230722_fedsda-v11-mnist4-seeds10-19`
- 基準版12の既定（研究ツール、製品と一致）: `fedsda-provisional-candidate-expert/20261023_uranium-v11-sample-holding-cross-evaluation-seeds0-9-n5000`、`…-mnist4-seeds10-19-n5000`
- 評価のclientの上限なし: `fedsda-provisional-candidate-expert/20261023_uranium-v12-all-evaluating-clients-*`
- 基準版13の候補: `fedsda-provisional-candidate-expert/20261023_uranium-v13-candidate-*`
- 候補の共有部の固定を外す（製品、基準版12）: `fedsda-component-ablation-v12/fedsda-without-candidate-validation`
- 拒否権: `fedsda-provisional-candidate-expert/20261023_uranium-v11-usage-evidence-veto-*`
- 案B1: `fedsda-provisional-candidate-expert/20261023_uranium-v12-soft-concept-sharing-*`（本規模）、`20261023_b1-*`（手元）
- 案C: `fedsda-provisional-candidate-expert/20261023_c-prediction-distillation-*`（手元）
- 基準版11のablation: `fedsda-component-ablation-v11/20261022_235734_core-components-v11-seeds0-9`

## 7. 前もって考えておいたこと（2026-10-23、主担当。再開のときに、そのまま使える）

### 7.1 残っている判断への推奨（決めるのはユーザー）
1. **基準版13には、2つとも入れるのを勧める。**
   - 評価させるclientの上限を外す: 精度の損が無く（−0.09〜+0.17pp）、計算は全体の+0〜1.7%、通信はほぼ同じ。旧で通信を抑えるために置いた固定の値3が、今の構成では理由を失っている。
   - 候補の共有部の固定を外す: CIRCLE2は2つの版（11・12）のablationで一貫して+0.1〜+0.2pp。ほかは±0.2ppでばらつき、MNIST4・500で−0.14pp（10 seed）。ただし13の候補（2つを合わせた形）のMNIST4・500は20 seedで+0.05ppで、系統的な悪化とは言いにくい。固定は、版7で候補検証を外したときに足した規則で、版10で見えた効果はMNIST4の崩れの偶然だった（ablationの文書）。規則を1つ減らす単純化として価値がある。
2. **次の研究は、IMPROVE-030の診断（手元で約1時間、決め手になる）を先に行い、並行してIMPROVE-029のE1a（研究ツールの包み1つ）を試すのを勧める。** 診断で「新しいモデルが僅差で負けている」ならS1へ、そうでなければIMPROVE-030のS3か、IMPROVE-029のE2（クラスに条件付けた重み）へ。
3. 先行研究の調査の報告（元のcheckoutの`docs/research/FedSDA案B1_B2の先行研究調査と新規性リスク評価.md`、Git管理外）は、worktreeの`docs/research/`へ取り込んでcommitするのを勧める（IMPROVE-028の12が参照し、論文の比較の計画——FedDrift・FedFomo・FedEM/FedRC・FedMerge——の根拠になる）。取り込むかはユーザーの判断。

### 7.2 基準版13のspecの設計の要点（手本は基準版12のspec）
- **評価させるclientの上限**: `runtime/fedsda_final_configuration_run_settings.py`の`build_final_configuration_fedsda_run_settings`は、実行条件（`experiment_run_conditions`。clientの数を持つ）を受け取るので、`maximum_evaluating_client_count_per_model=_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL`（3）を、`experiment_run_conditions.client_count`に替えるのが最も簡単（設定の型を変えず、「上限なし」をNoneなどで表す必要が無い）。研究ツールの変種`product-default-all-evaluating-clients`も、同じく参加者の設定をclientの数に置き換えている。
- **候補の共有部の固定**: 同じ関数の`candidate_shared_feature_update_policy="keep_shared_features_fixed"`を`update_shared_features_with_candidate`へ替える。すると、構成`fedsda-without-candidate-validation`（候補検証なし＋共有部も更新）が、既定と同じ設定になる。goldenの事例（`BASELINE_CONFIGURATION_CASE_NAMES`）にこの構成があるので、(a) 名前を残して既定と同じになることを受け入れる、(b) 固定する構成（例: `fedsda-fixed-candidate-shared-features`）をablationとして足す、のどちらかをspecで決める（(b)を勧める。今の既定から外した要素をablationで測れるようにするのが、これまでの形）。
- **版1〜12の照合**: goldenの`behavior_golden_cases.py`で、参加者の設定の`maximum_evaluating_client_count_per_model`（版12まで3）と、clientの設定の`candidate_epoch_training_settings.candidate_shared_feature_update_policy`（版7〜12で`keep_shared_features_fixed`）を、版ごとの「既定の値が変わった項目」の表（`_CLIENT_SETTING_VALUES_CHANGED_AFTER_BASELINE_VERSION`と、参加者の設定の同じ形の表——無ければ足す）で戻す。候補の共有部の扱いは、版1〜6は`update_shared_features_with_candidate`（今の表の`_CANDIDATE_SHARED_FEATURE_UPDATE_POLICY_DEFAULTS`）なので、版7〜12だけが固定であることに注意。
- **構成`fedsda-with-candidate-validation`**: これまでの版と同じく、今の扱い（上限3、共有部の扱いは今の値）を明示するのが安全（候補検証との組合せは未検証）。
- **研究ツールの基点の固定**（`_execute_job`）: `product-default`系以外の変種の置き換えに、上限3と、共有部を固定する扱いを足す。
- **研究ツールとの照合**（既定を替える前に）: 製品の既定（基準版12）に2つの項目だけを新しい値にした全体runと、研究ツールの`product-default-all-evaluating-clients-candidate-shared-feature-update`が、4データ以上で一致すること（精度・通信量——メッセージの数を含む——・予測の記録と最終のパラメータの要約・計算回数）。
- **版13の必須の経路（案）**: クロス評価で、1つの（候補、対象）の組を4つ以上のclientが評価した記録があるケースが少なくとも1つあること（`client_cross_evaluation`の記録を、同期・候補・対象ごとに数える）。上限を外したことが実際に効いている証拠になる。候補の共有部の更新は、候補の採用があるケース（版7からの採用の経路のケース）で起きるので、今の必須の経路（採用・切替え・再較正）で守られる。
- **testの絞り方**: 版12と同じく、重要なもの（研究ツールとの照合、版1〜12の照合、版13の必須の経路、版12→13の差の説明——FedDrift・FedAvgと候補検証ありの構成のケースは同じ値）に絞る。

