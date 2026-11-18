# Research Log

## 調べたこと
- 単一ストリームの既存手法の条件（`references/surveys/単一データストリームにおけるオンライン分類・再来概念ドリフト手法の文献調査.md`）: ARF、SRP、PEARL、DriftMoEなどはHoeffding tree、RCD・CPF・ECPF・GroCHなどは学習器を差し替えられる枠組み。MOAの`EvaluatePrequential`とRiverの`progressive_val_score`は、最初の標本から予測→学習で評価し、別の準備の区間を持たない。調査は、提案手法の学習器をHoeffding treeに替える比較（学習器に依らないことの証拠）と、事前学習の500件の非対称の解消を勧めている。
- 論文の手順の正本: `FedHASP-paper/ja/sections/04-method.tex`（4.2柔らかい予測、4.3固い割り当て）。
- 製品の1標本の処理（`runtime/observed_sample_processing.py`）: 予測→重みの更新→監視→保留へ追加→警報の判定（MDLの選択、min(n,k)<5なら待つ）→警報なら割り当てと監視の始め直し、警報でなければ最古の標本を割り当てて学習。神経網の分類器・共有の特徴抽出部・保有モデルの登録簿を直接扱い、学習器を差し替える口が無い。
- 判断の部品はtorchを使わない: `overall_and_true_class_loss_monitoring`、`loss_change_detection_settings`、`fixed_share_weights`、`prediction_combination_settings`、`minimum_description_length_assignment_selection`、`loss_baseline_selection`、`model_and_class_loss_statistics`、`bounded_loss_moments`。WSLのRiver環境（`~/river-env`、Python 3.12、river 0.27.0、torchなし）で`PYTHONPATH=src`にしてimportでき、`torch`が読み込まれないことを確かめた（2026-11-18）。
- 損失の統計: 製品は、標本をモデルへ割り当てるとき、そのモデルの（学習前の）監視用の損失を`record_assigned_loss`で足す（`runtime/assigned_training_sample_absorption.py`）。監視の基準値は、現在のモデルの統計から`select_loss_monitoring_baseline_mean_loss`で選ぶ。警報の後は`OverallAndTrueClassLossMonitor.reset(baseline_loss_mean=...)`で始め直す。
- MDLの新しいモデルの符号長: 製品は「標本を順に予測し、その標本で1回学ぶ」学習者の対数損失の和（既定は逆順の1本の流れで、1件刻みの全k）。木の`predict_proba_one`→`learn_one`と同じ形。
- 系列の入力: 研究ツール`stream_export`のNPZ（`features`、`labels`、`concept_ids`、ほかに事前学習の標本）。Riverの比較手法も同じファイルを読む（`river_baselines.py`）。
- 実行環境: Windowsでは新しいDLL（river_rust ほか）の読み込みが拒否されるため、Riverを使う処理とtestはWSLの`~/river-env`で行う。

## 判断
- 製品の中で学習器を差し替えず、別のパッケージで書く: 製品の流れは神経網の部品に結びついていて、差し替えには連合学習の経路も含めた作り替えが要る。ユーザーの指示（後で切り離しやすく）にも合う。
- 判断の部品は製品のコードを使う（写さない）: 同じアルゴリズムであることを部品の共有で示せる。切り離すときは、その部品を写せばよい。
- 製品の全体の流れとの出力の照合はしない: 学習器が違うので出力は一致しない。神経網の学習器をつなぐ口を作れば照合できるが、ユーザーとの合意で、単体の実装にして接続できる部分で接続する形にした（2026-11-18）。代わりに、手順の対応表（design.md）と、流れのtestで確かめる。
- 学習は割り当てた時点の1回: 既存手法の条件（標本ごとに1回）に合わせる。製品の再生のミニバッチは神経網のための仕組み。
- 新しいモデルの初期値をMDLで短かった学習者の種類にする: 木は数エポックの学習をせず、MDLで測った学習者と同じ学び方になる。木は過去のコンセプトで作った分岐を捨てられないので、別のコンセプトの保有モデルの写しから始めるのは不利。
- ウォームアップ200件: 事前学習を使わないので、監視の基準値を作るための区間。手法の中の規則として扱い、予測は評価に含める（2026-11-18ユーザーと合意）。値はRiverのHoeffding treeの既定のgrace periodと同じ。
- 確率の平滑化0.01: 木は確率0を出しうるので、MDLの対数損失を有限にする。一様分布との混合なので最大のクラスは変わらない。
- 帰属の保留はこのパッケージで小さく持つ: 製品の保留の部品は、製品の標本の型（torchのTensor）と組になっているため。規則（容量、最古を割り当て、警報で空にする）は同じにする。
