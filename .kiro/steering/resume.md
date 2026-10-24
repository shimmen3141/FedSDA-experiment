# リファクタリング再開案内

2026-10-24更新。エージェント間の引継ぎ用。承認・進捗・検証証拠の正本は各spec。この文書は現在地への入口とする。

## 現在地

- 共有部接続のuranium結果（2026-10-24）: [本規模の検算と判断](../../docs/research/fedsda-shadow-reconnection-uranium-results-20261024.md)。120条件・240 run成功、全250ソース一致、計測なし採用方式と全120対で秒以外の全保存値・MAC・採用記録・全654561トレース行一致。488採用中、直近348悪化/2同値/138改善、将来282/2/204。再接続による損失悪化は確認できたが、全体精度低下の主要因は未確定。推奨は接続後の影と現行モデルを同じ過去標本で比較する手元対照の準備。本規模来歴追跡は別出力で並行できる。今回追加実験なし・製品変更なし。

- 採用モデルの追跡（2026-10-24）: [登録・送信・統合・配布・予測係数の非介入計測](../../docs/research/fedsda-shadow-model-lifecycle-20261024.md)を追加。6入力600標本の12 run成功、元採用方式と全6対で秒以外の全保存指標・MAC・採用経路一致。9採用中5件は終了時も未登録保留、4件は登録と同じ同期で統合（3件は最終同期）。小規模の頻度を本規模へ一般化しない。共有部接続のuranium実験はユーザーが開始済み。回収後に両経路を分けて評価する。製品変更なし。

- 共有部接続の計測（2026-10-24）: [非介入の計測](../../docs/research/fedsda-shadow-feature-reconnection-20261024.md)を追加し、専用3件成功、Luna実装レビューPASS。6入力600標本の12 runが成功し、以前の採用方式と全6対で秒以外の保存値・差引後MAC・採用経路一致。9採用すべてに共有部差、直近損失は6悪化/3改善。主要因とは未確定で、次はuraniumの同じ120条件で計測する推奨。参照照合は同じ環境の結果を使用。製品変更なし。

- ②のuranium結果（2026-10-24）: [採用120条件・240 runの結果](../../docs/research/fedsda-shadow-adoption-uranium-results-20261024.md)を回収・検算。失敗0、249ソース一致、全baselineが旧サーバと秒以外一致。採用488件。SEA4全体差+0.0164/−0.0194pp、目標の概念2は−0.3203/−0.0340pp。SINE2通常は+0.0448/+0.1418ppだがseed差と追加費用あり。無変化8条件は平均低下。現行形の本実装は推奨せず、採用時の共有部再接続と採用モデルの登録・統合経路の小規模計測が次の推奨。製品変更なし。

- ①の拡張と雑音対照（2026-10-24）: [5,000標本・4 seedの24 runと雑音なし8 run](../../docs/research/fedsda-shadow-optimizer-expanded-20261024.md)を手元で完了、失敗0。各データ条件内で精度差0pp・通信不変。概念2の同時D/C交差はoptimizer継続/resetとも3/24で主因を支持せず、全体39件の同時交差は全て再反転。無雑音・無変化でも交差が残る。単純なC条件の追加採用は見送る。次の推奨はユーザーがuraniumで開始した②の実際の精度・費用の確認。製品srcは変更なし。

- S1の次段階（2026-10-24）: ユーザー指示により[採用実験②とoptimizer対照①](../../docs/research/fedsda-shadow-adoption-and-optimizer-followup-20261024.md)を並行する。②はuranium、①は手元。製品srcは変更しない。②のツールはpush・コマンド提示済み（サーバ実行はユーザー担当）。①は手元の12条件・24 runを完了、精度差0pp・通信不変。D/C同時交差はoptimizer継続17件、リセット16件で主因とは断定できず、概念2の影は1件だけ。次の推奨は②の本規模の精度差・費用の確認。

- 標準指標の一覧化（2026-10-24）: S1対照の120条件から、精度・上下通信・MAC・経過時間と診断を併記する[一覧と方法](../../docs/research/fedsda-shadow-learner-s1-controls-uranium-results-20261024.md)を追加。analysis_standard_metrics.md、standard_metrics_by_seed.csv、summary_standard_metrics.jsonを元結果のディレクトリへ保存。今後の研究報告は毎回標準指標を明示し、次の候補と推奨も添える。次の推奨は引き続き初期化・optimizer履歴の対照設計。再実験はしていない。

- uraniumのS1対照の結果確認（2026-10-24）: [120条件・240 runの分析](../../docs/research/fedsda-shadow-learner-s1-controls-uranium-results-20261024.md)を記録。失敗0、全120対で既定と保存値一致・追加計算を除く計数一致。概念2でD/C同時交差は間隔100が13件、500が16件、各6 seedだが無変化でも発生し、D交差後の再反転も多い。D>0だけで採用する案は保留。推奨の次段階は初期化・optimizer履歴の小規模な対照設計。元結果はresults/fedsda-shadow-controls/20261024_uranium-s1-controls-seeds0-9-n5000。以後の報告には次の候補と推奨を添える。

- サーバ結果待ちの追加作業（2026-10-24）: [S1の逆転後の安定性と分析方針](../../docs/research/fedsda-shadow-learner-s1-stability-20261024.md)を作成。既存8 runを再学習せず分析し、概念2の8交差中6件が概念維持中に非正へ戻る一方、同概念の最終観測は8件とも正と確認。seed0件も含む集計・打切り・D/Cの安定性を調べる追加分析CLIを準備。サーバの途中結果は未参照。実験を開始する既存のCLI・診断は変更していない。

- 実験環境の最新訂正（2026-10-24）: uraniumは256コア・標準64 workers。手元16コア向けの14/各7をサーバの上限としない。CLIの案内と古い記述の所在は[tech.md](tech.md#研究実験の実行環境とコマンド2026-10-24更新)。今回のS1対照も`--workers 64`で案内する。

- 2026-10-24の次段階: [S1の無変化・オンライン継続対照](../../docs/research/fedsda-shadow-learner-s1-controls-20261024.md)を準備。6入力・seed1・600標本の対照12 runが成功、関連60件成功、Luna medium実装レビューPASS。製品変更・採用は無し。本規模は6入力×間隔100/500×10 seedの120条件（既定と診断で240 run）。uraniumでの転送・実行はユーザー担当、コマンドは研究記録。ユーザーが研究ツールのpushを許可。uraniumはrefactor/architectureをpullして専用shellで実行する（ZIPは予備）。本規模は結果回収・照合・分析完了。最新の判断は上記のサーバ結果の記録。

- 続く2026-10-24の研究: ユーザー指示で[S1の成立性診断](../../docs/research/fedsda-shadow-learner-s1-diagnostic-20261024.md)を実施。影を予測・採用に使わず損失差だけ追うツールを作成し、境界修正・Luna再レビュー後の5,000標本・4 seed・8 runは失敗0で完了（20261024_s1-feasibility-fixed-n5000）。SEA4概念2は24影中8件が概念変化前に初期差を回収、うち6件はseed1。他概念でも交差し、採用機能は未試作・未採用。次の研究候補は無変化対照・同じ学習手順の対照の準備。共通保存項目と追加計算を除く計数は既定と全8対一致。専用全体対照2件＋境界修正後関連55件成功、Luna medium PASS。研究ツールなのでspecは作らず、製品srcは変更していない。

- 2026-10-24の研究作業: [E1a](../../docs/research/fedsda-mixing-past-posteriors-e1a-20261024.md)と[E2](../../docs/research/fedsda-class-conditioned-prediction-e2-20261024.md)の検証を完了。E2は短い予備対照に加えてMNIST4の5,000標本・4 seedまで確認したが、改善が小さく専用枝単独は低下したため、現行形は未採用・本規模化見送り。製品は基準版12のまま。検証・レビュー・数値と採否の正本は各研究記録。E1aの原因の時系列診断、E1b/E3の試作は未実施。研究ツール段階ではspec不要、本実装を決めたらcc-sddというユーザー指示を維持する。1時間以上かかる見込みの実験は55分おきに確認し、早く完了通知が来たらその時点で分析する。

- 作業場所は`.worktrees/refactoring`、ブランチは`refactor/architecture`。mainへはまだマージしない。同じworktreeの主担当は一人にする。
- **研究の現在地と次の一手は[研究の引継ぎ（2026-10-23）](../../docs/research/research-handoff-20261023.md)を最初に読む**（前提、ユーザーとの分担の取り決め、分かったこと、残る判断、次のtask）。
- 直近に完了したspecは[fedsda-cross-evaluation-clients-with-target-samples](../specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)。対象の標本を返せるclientだけをクロス評価させる扱いを実装し、既定を切り替えた（基準版12。goldenは生成元`fab7f040`、定数2つは12）。specは完了。制約・検証の正本は対象specのImplementation Notes。開始時にはGitの実状態を再確認する。
- FedSDAと比較構成、FedDrift、単一モデルのFedAvg（旧Oblivious）の移植・全体run・成果物・掃引入口は接続済み。旧実装は参照として残っている。古い移植順を次の作業と誤認しない。
- FedSDAの既定は配布時モデルとローカル更新モデルの混合予測、帰属保留FIFO30、共通時間尺度T120。Tはモデル別重みと二者混合で共通に使い、両者を別パラメータにしない。
- 2026-10-20から、既定は候補を検証せずにその場で採用し、候補の学習では共有部を固定する（基準版7。[spec](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)）。それまでの候補検証ありは構成`fedsda-with-candidate-validation`、検証なしで共有部も更新する構成は`fedsda-without-candidate-validation`。候補検証の処理は残している（消すかは、ユーザーが別に決める）。
- 2026-10-20から、クロス評価で配布するモデルを使う（基準版8。[spec](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)。精度は同じで、FedSDAの下りの通信量だけが減る。それまでの結果の通信量は旧の扱い）。
- 2026-10-20から、統合で同じ共有部を平均せずに保つ（基準版9。[spec](../specs/fedsda-consolidation-keeps-shared-features/tasks.md)。統合が起きたrunでは値がわずかに変わる。版8のSEAの通信量の増加は、この誤差による共有部の送り直しだった）。
- 2026-10-21から、警報の後の変化点と帰属先を最小記述長の原理で選ぶ（基準版10。[spec](../specs/fedsda-minimum-description-length-alarm-interval-assignment/tasks.md)。設定`alarm_interval_assignment_policy=minimum_description_length`）。保有モデルの再利用の判定に固定の許容量γを使わない。構成`fedsda-with-candidate-validation`は今の扱い（`detector_change_span_and_mean_loss_increase_tolerance`）のまま。サーバのクラスタリングの閾値γ=0.1は残る（[IMPROVE-026](../../docs/research/improvement-candidates/improve-026-server-clustering-by-description-length.md)）。SINE2・CIRCLE2・間隔100の通信の増加（見逃した再利用）は、このspecでは扱っていない。
- 2026-10-22から、「新しいモデル」の仮説を、初期化し直した学習者とwarm startの学習者の短い方＋ln 2で測り（`new_model_hypothesis_learners=reinitialized_and_warm_started_learners`、NEW-006）、初期モデルの損失の統計を事前学習の標本の5分割交差検証で作る（`initial_model_loss_statistics_estimation=five_fold_cross_validation`、LEGACY-020）（基準版11。[spec](../specs/fedsda-warm-started-learner-and-cross-validated-initial-statistics/tasks.md)）。構成`fedsda-with-candidate-validation`は基準版10までの扱い。研究ツールの変種は、`product-default`以外、基準版10の既定を基点にする。
- 2026-10-23から、クロス評価で、対象のモデルを保有し、その標本を返せるclientだけを評価させる（基準版12。`cross_evaluation_client_selection_policy=clients_with_target_model_samples`、LEGACY-022の修正）。構成`fedsda-with-candidate-validation`は今の扱いを明示する。研究ツールの変種は、`product-default`系以外、基準版10の既定を基点にする。
- 動作基準は基準版12・環境別golden。版12は[fedsda-cross-evaluation-clients-with-target-samples](../specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)。版6は[fedsda-default-synchronized-local-mixture-prediction](../specs/fedsda-default-synchronized-local-mixture-prediction/tasks.md)、版7は[fedsda-default-adoption-with-fixed-candidate-shared-features](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)、版8は[fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)、版9は[fedsda-consolidation-keeps-shared-features](../specs/fedsda-consolidation-keeps-shared-features/tasks.md)、版10は[fedsda-minimum-description-length-alarm-interval-assignment](../specs/fedsda-minimum-description-length-alarm-interval-assignment/tasks.md)、版11は[fedsda-warm-started-learner-and-cross-validated-initial-statistics](../specs/fedsda-warm-started-learner-and-cross-validated-initial-statistics/tasks.md)、環境と手順は[refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)。環境差だけを理由にgoldenを更新しない。

## 次の一手

1. [CLAUDE.md](../../CLAUDE.md)、[AGENTS.md](../../AGENTS.md)、[共通引継ぎ手順](agent-handoff.md)を読み、ブランチ・HEAD・差分・対象specを照合する。Claude版cc-sddは`.claude/skills/`、仕様正本は共通の`.kiro/specs/`。
2. 次の一手は[研究の引継ぎ](../../docs/research/research-handoff-20261023.md)の5（基準版13に入れるものと、次の研究の方向について、ユーザーの判断を仰ぐ。判断の後に基準版13のspec）。案B1・案Cは試して採らなかった（IMPROVE-028の14・16）。次の研究の候補は、予測の混合を強くする案E（[IMPROVE-029](../../docs/research/improvement-candidates/improve-029-stronger-prediction-mixture-for-recurring-and-partial-concepts.md)）と、SEA4の概念2をclientの側で分ける案（[IMPROVE-030](../../docs/research/improvement-candidates/improve-030-split-partially-changed-concept-on-client.md)）。どちらも実装の手順まで書いてある。候補検証研究を続ける場合は下の研究記録を読み、DやPを未実験の案として再実装しない。
3. 次の候補は、改善の4案（[IMPROVE-020〜023](../../docs/research/improvement-candidates/README.md)。進める順は020→022→021→023、各ファイルに前提・方針・やること）。020は採用し、基準版8として実装した。022（候補の学習の早期終了）の手元の比較を2026-10-20に始めた。新しい既定の下でのablationは、当面konanの版6の結果で代える。候補検証の処理を消すかは、ユーザーの判断を待つ。
4. 新しい研究対照は独立ツールへ閉じ、製品経路を変える場合は別specと実装前の独立仕様レビューを経る。現在の既定を黙って変更しない。

## 決定事項

- 研究目標は候補検証の削除自体ではなく、「検証なしの改善を残し、検証ありが防ぐ低下・通信増加をより単純に抑える」こと。検証あり＋新方式も候補。許容精度低下幅は未設定で、自動採否はしない。
- D（待機後強制採用）、R（既存重み比率保持）、P（新IDの初期質量抑制）、共有部25%反映は、一般的な置換方式として採用していない。候補学習中の共有部の固定は、候補検証ありとの組合せ（codexの小規模・研究ツールの本規模とも不安定）では採らず、検証なしとの組合せを2026-10-20に既定にした（本規模・seed 0〜9）。条件限定の改善と一般的な優位を区別する。
- 探索用の対照は`tools/research_*`等へ隔離し、共通アルゴリズム・設定選択肢・通常pytest/CI/freshへ枝を増やさない。研究専用checksは明示実行する。通常入口へ採用済みの再開補助・メモリ回収・seed構文は、この研究分岐の扱いとは区別する。
- 標準JSON/NPZ/READMEの形式は再開補助・省メモリ化・seed構文のためには変更していない。研究診断用の別形式JSONは標準run形式と混同しない。
- 旧実装との一致を守る参照設定と、改善した新既定を区別する。旧結果の一括変換ではなく、新実装でablationを実行する方針を維持する。
- レビュー手順は共通引継ぎ手順を参照する。全pytest、および権限で実行不能なfresh smokeは、独立レビュワーが主担当の実測証拠を照合し、独立未実行を明記して判定できる。過去の子エージェントIDは来歴であり、Claudeから再利用する接続先ではない。

## 研究記録の入口

- [要素ablation](../../docs/research/fedsda-component-ablation-20261017.md): 720runの条件別比較。検証の有無はデータ・間隔で符号が変わる。「ローカル更新モデルのみ」と「現在の学習帰属モデルのみ」は別方式。
- [候補検証の原因診断](../../docs/research/fedsda-candidate-validation-diagnosis-20261017.md): MNIST2/4、共有部置換、予測重みなどの初期切分け。
- [非MNIST共有部反映](../../docs/research/fedsda-candidate-backbone-blend-nonmnist-20261018.md)、[候補共有部固定144run](../../docs/research/fedsda-frozen-candidate-backbone-20261018.md): 単純な共有部保護だけでは全条件を改善できない。後者の間隔200は現行処理どおり2400標本評価（生成2500）。
- [単純化の整理](../../docs/research/fedsda-candidate-validation-simplification-20261018.md): 問題の分解と次の仮説。5seed時点の表なので最新のV/O判断は次の10seed報告も参照する。
- [サーバ3実験・10seed比較](../../docs/research/fedsda-server-results-20261018.md): 新規660runと既存対照の受領・照合。検証なしはMNIST2の全間隔で平均低下し、MNIST4・500では改善。通信は24条件平均すべて増加。混合予測とFedDrift/FedAvg比較も含む。
- [D本規模54run](../../docs/research/fedsda-candidate-waiting-diagnostic-20261018.md): 待機後強制採用はV/O両方を上回る代替にならず、共通経路から分離済み。
- [R/P本規模72run](../../docs/research/fedsda-prediction-initial-mass-n5000-20261019.md): 予測初期質量だけの制御では保護効果を代替できない。学習・適応・通信をOと揃えた対照。[実行手順](../../docs/experiments/prediction-initial-mass-diagnostic.md)。
- [uranium回収対照](../../docs/research/fedsda-live-reclamation-uranium-20261019.md): 学習結果を変えずにメモリを減らす実測と適用範囲。現在は通常FedSDAへ接続済み。

## 採用済みの実行補助と環境の注意

- 掃引の再開: 同条件の過去出力と進捗から再開先を選択できる。完全な既存runはスキップし、完了済みを選ぶと実行を終える。中断runは補助証明と条件・コード・環境が一致する場合だけ退避して再実行する。証明のない旧不完全ファイルや破損・不一致を盲目的に上書きしない。[手順](../../docs/experiments/experiment-sweep.md)。
- seed: `--seeds 0-9`は両端込み。`--seeds 0-2 5 8-9`の列挙混在も可。負数・逆順・重複・範囲重複は拒否する。間隔/T/FIFOのステップ付き範囲やdatasetの`all`指定は未実装。
- メモリ: Linux/glibcの通常FedSDA全体run・掃引は、正常な同期末に`malloc_trim(0)`を呼ぶ。Windows等の非対応環境はno-op。FedDrift/FedAvgや低レベルstream単独には接続していない。[運用と制限](../../docs/experiments/fedsda-runtime-memory-reclamation.md)。
- uraniumでのMNIST2・seed0・1workerの1対では最大RSS81.54%減、時間1.79%増、全指標・133配列等一致。32/64workersでの導入後の同条件対照は未実施。この削減率から並列数をそのまま算出しない。アルゴリズム・標本表現・allocator設定の一律変更はしていない。
- 過去の独立回収prototypeは旧source hash固定の研究再現用。通常実行は最新ブランチを取得する。uraniumが一時detached HEADだったためブランチ復帰手順を案内済みだが、サーバ側更新の完了は未確認。
- サーバ実行と転送はユーザー担当（踏み台経由）。長時間実験はLinux `.venv`・tmux、各数値ライブラリ1thread。uraniumのコア数だけでworkerを増やさず、メモリ・負荷を確認する。
- Windows検証Pythonは元checkoutの`venv/Scripts/python.exe`。Linux検証WSLは`Ubuntu`を明示する（既定がdocker-desktopになっていた）。MNIST入力・固定環境は基準文書に従う。
- 研究入力・生の結果・一時スクリプトの多くはGit管理外。`results/`が存在するこのworktreeと、Gitだけをcloneした環境は同じではない。各報告の入力パスと完全性を確認し、一括stageしない。

## 未解消の事項・次の候補

- 候補検証を単純化しV/O両方に優位となる方式として、検証なし＋候補の共有部の固定を採用した（[研究記録](../../docs/research/fedsda-provisional-candidate-expert-20261019.md)）。検証なしのMNIST2の低下が共有部の上書きから来るという説明は、固定で低下が消えたことと合うが、直接の切り分けではない。
- 純粋予測方式seed5〜9の240run、手法比較（間隔100/500）seed5〜9の180runは補完コマンドを提示済みだが、実行・結果受領は未確認。確認なしに完了扱い・再発注しない。
- LEGACY/IMPROVE/NEW/UNPORTEDの現在の採否は[実装findings](../../docs/research/implementation-findings/README.md)と[改善候補](../../docs/research/improvement-candidates/README.md)が正本。10/13以降の修正・採否を、古い「残るLEGACYをまとめて判断」へ戻さない。
- LEGACY-001とLEGACY-014の概念計数は旧実装を外した後に直す判断、NEW-004は旧依存の計算量照合を外す準備の記録。LEGACY-019の改名参照除外の頻度・影響は未確認。詳細と変更時期は各記録を参照する。
- 未移植の細目はUNPORTED-001（検出episode）、002（参照も学習する検証）、003（回復期限定の混合）。現在の不具合や移植漏れと自動判定しない。その他の診断追加や旧実装削除は、目的・依存を確認してから仕様化する。
- 一部の古いspecはtask完了でもphaseが`implementation-complete`等のまま。履歴上の状態表現と直近specの未完了を混同せず、spec.jsonとtasksの証拠を読む。
- 動作図HTMLは別ブランチ`docs/fedsda-processing-flow`で整理・push済み、専用worktreeは削除済み。mainへのマージは保留。refactoringへ自動コピー・混在させない。

## 直近・主要な完了spec

- [fedsda-consolidation-keeps-shared-features](../specs/fedsda-consolidation-keeps-shared-features/tasks.md): 統合で同じ共有部を平均せずに保つ扱いの既定と基準版9。
- [fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md): クロス評価で配布するモデルを使う扱いの既定と基準版8。
- [fedsda-default-adoption-with-fixed-candidate-shared-features](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md): 候補検証なし＋候補の共有部の固定の既定と基準版7。
- [seed-range-command-arguments](../specs/seed-range-command-arguments/tasks.md): seedの両端込み範囲と列挙混在。
- [fedsda-runtime-memory-reclamation](../specs/fedsda-runtime-memory-reclamation/tasks.md): 通常FedSDAの同期末メモリ回収。
- [run-memory-reclamation-prototype](../specs/run-memory-reclamation-prototype/tasks.md): 同期末回収の独立対照ツール。
- [prediction-initial-mass-diagnostic](../specs/prediction-initial-mass-diagnostic/tasks.md): R/Pの独立研究診断。
- [candidate-validation-waiting-diagnostic](../specs/candidate-validation-waiting-diagnostic/tasks.md): 待機後強制採用Dの研究診断。
- [experiment-sweep-resumption](../specs/experiment-sweep-resumption/tasks.md): 過去出力の選択・再開・安全復旧。
- [fedsda-component-ablation-analysis](../specs/fedsda-component-ablation-analysis/tasks.md): 要素ablation解析と人による採否判断の補助。
- [fedsda-default-synchronized-local-mixture-prediction](../specs/fedsda-default-synchronized-local-mixture-prediction/tasks.md): 混合予測・FIFO30・共通T120の既定と基準版6。
- [fedavg-experiment-run](../specs/fedavg-experiment-run/tasks.md): 単一モデル比較手法FedAvgの移植。
- [feddrift-experiment-sweep](../specs/feddrift-experiment-sweep/tasks.md): FedDriftの全体run・成果物を掃引入口へ接続。
- [important-behavior-regression-goldens](../specs/important-behavior-regression-goldens/tasks.md): 新実装の重要経路を環境別goldenで固定。

上記は直近と主要な到達点の索引で、全specの一覧ではない。以前の移植は[roadmap](roadmap.md)と各specを参照する。完了specのレビュー経緯・検証件数・commitは各tasks.mdのImplementation Notes等へ集約し、ここへ複製しない。
