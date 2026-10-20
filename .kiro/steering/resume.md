# リファクタリング再開案内

2026-10-20更新。Claudeへの再引継ぎ用。承認・進捗・検証証拠の正本は各spec。この文書は現在地への入口とする。

## 現在地

- 作業場所は`.worktrees/refactoring`、ブランチは`refactor/architecture`。mainへはまだマージしない。同じworktreeの主担当は一人にする。
- 直近の対象[fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)は完了。下記の最近のspecに未実装task・未解消レビュー指摘はない。開始時にはGitの実状態を再確認する。
- FedSDAと比較構成、FedDrift、単一モデルのFedAvg（旧Oblivious）の移植・全体run・成果物・掃引入口は接続済み。旧実装は参照として残っている。古い移植順を次の作業と誤認しない。
- FedSDAの既定は配布時モデルとローカル更新モデルの混合予測、帰属保留FIFO30、共通時間尺度T120。Tはモデル別重みと二者混合で共通に使い、両者を別パラメータにしない。
- 2026-10-20から、既定は候補を検証せずにその場で採用し、候補の学習では共有部を固定する（基準版7。[spec](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)）。それまでの候補検証ありは構成`fedsda-with-candidate-validation`、検証なしで共有部も更新する構成は`fedsda-without-candidate-validation`。候補検証の処理は残している（消すかは、ユーザーが別に決める）。
- 2026-10-20から、クロス評価で配布するモデルを使う（基準版8。[spec](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)。精度は同じで、FedSDAの下りの通信量だけが減る。それまでの結果の通信量は旧の扱い）。
- 動作基準は基準版8・環境別golden。版6は[fedsda-default-synchronized-local-mixture-prediction](../specs/fedsda-default-synchronized-local-mixture-prediction/tasks.md)、版7は[fedsda-default-adoption-with-fixed-candidate-shared-features](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)、版8は[fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)、環境と手順は[refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)。環境差だけを理由にgoldenを更新しない。

## 次の一手

1. [CLAUDE.md](../../CLAUDE.md)、[AGENTS.md](../../AGENTS.md)、[共通引継ぎ手順](agent-handoff.md)を読み、ブランチ・HEAD・差分・対象specを照合する。Claude版cc-sddは`.claude/skills/`、仕様正本は共通の`.kiro/specs/`。
2. 候補検証研究を続けるなら、下の研究記録を読む。DやPを未実験の案として再実装しない。現在、次の製品アルゴリズム変更のspec・採用決定はない。
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
