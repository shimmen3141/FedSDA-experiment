# Research Log

## 調査の範囲
- 既存の構成への拡張（軽い調査）。接続点と、論文のアルゴリズムの細部。

## 論文（Kolter & Maloof, JMLR 8, 2007、2755–2790）
- 図1（2764頁）: 専門家ごとに予測→（i mod p = 0 かつ誤り）なら重みにβを掛ける→重みを予測クラスへ足す→最大のクラスを全体の予測→（i mod p = 0 なら）正規化→θ未満の削除→全体の予測が誤りなら重み1の専門家を追加→全専門家を学習→全体の予測を出力。
- 3節（2763〜2765頁）: 正規化は最大の重みが1になるよう一様に縮める（新しい専門家が予測を支配しないため）。最後の専門家は消さない。pの間は重みの更新と専門家の追加・削除をしないが、学習はする。最初の専門家は「既定のクラスか、前の経験・背景知識で予測してよい」。
- 4.1節（2767頁）: STAGGERではβ=0.5、θ=0.01、p=1。4.2節（2772頁）: 大きく雑音のある問題（SEA、5万標本、10%のクラスの雑音）ではβ=0.5、θ=0.01、p=50。今回の系列は5,000〜50,000標本で、SEAを含むので、p=50を採る。
- 論文に書かれていない点: 投票の同点の扱い（最小のクラスにする）。予測を出さない専門家の扱い（神経網は常に予測するので起きない）。
- 論文の評価は、各時刻の後に別のテスト集合で測る。今回は予測→学習の評価なので、図1のままでは、全体の予測が同じ標本のラベルで減衰した重みに依る（ラベルの漏れ）。評価の予測は学習の前の重みで行い、学習の手順（追加の判断に使う投票し直した予測を含む）は図1のとおりにする。
- 公式の実装（Java、WEKA、`dwm.jar`の`dwm/src/java/guml/classifiers/DWM.java`、Revision 1.12、2008-06-28。ユーザーが2026-11-17にMaloofのページから取得し、元のcheckoutの`references/code/DWM/dwm.jar`（`C:\Users\yshin\Research\FedSDA-experiment\references\code\DWM\dwm.jar`。worktreeには無い）に置いた。中身はJavaのソース4つと説明のHTML・`build.xml`・README・GPL v3の`COPYING`で、コンパイル済みのクラスと`META-INF`は無い。論文のPDFは`references/papers/DWM/kolter07a.pdf`（主担当がJMLRから取得したものとSHA-256が一致）。実行はせず、jarを展開してソースを読んだ）:
  - `updateClassifier`（学習）: `epochs++`の後、専門家ごとに`distributionForInstance`→`Utils.maxIndex`で予測クラス→`epochs % period == 0`かつ誤りなら重みに`beta`を掛ける→その重みを予測クラスへ足す→最大の重みを控える。全専門家の後に`Utils.maxIndex`で全体の予測。`epochs % period == 0`なら、控えた最大の重みで全重みを割り（`scaleWeights`）、θ未満を消し（`removeExperts`）、全体の予測が誤りなら新しい専門家を重み1で足す（上限`maxExperts`に達していれば最も弱い専門家を消してから。既定の上限は無制限）。最後に、足したばかりの専門家を含む全専門家を`updateClassifier`で学習する。図1と同じ順。
  - `distributionForInstance`（予測）: 今の重みで各専門家の予測クラスへ重みを足して正規化するだけで、重みを変えない。WEKAの予測→学習の評価ではこれが学習の前に呼ばれるので、評価の予測は学習の前の重みになる。この設計の扱い（Requirement 2.1）と同じ。
  - 同点: `Utils.maxIndex`は最初の最大の位置を返すので、クラスの番号が最小のもの。この設計と同じ。
  - 新しい専門家: `Classifier.makeCopy(m_Classifier)`を空のデータで`buildClassifier`した、何も学習していない学習器。この設計は、事前学習の直後の神経網の複製にする（違う点）。論文3節は最初の学習器について「既定のクラスか、前の経験・背景知識で予測してよい」とし、`fedavg`とFedHASPの出発点も事前学習の初期モデルなので、学習器をそろえるためにこの扱いを採る。
  - βとθの検査は開区間(0, 1)、pは1以上。この設計の値（0.5、0.01、50）は範囲内。

## 既存の実装との対応
- `fedavg`のclient（`methods/fedavg/fedavg_client.py`）: 構造`independent_models_without_shared_backbone`、全到着標本の保持、保持数がbatch（32）以上なら標本ごとに`python_random_generator.sample(保持, 32)`で1回`update_independent_model_parameters`。予測は2クラスで`scores > 0.5`、それ以外は`argmax`。DWMの専門家は同じ部品を同じ順で呼ぶ。FedAvgのclientはさらに損失の統計のためのforwardをするが、パラメータと乱数には影響しない。
- 掃引への接続は`sweep/experiment_configurations.py`（構成の登録と設定の組立て）、`sweep/experiment_sweep_plan.py`（設定の型の和）、`sweep/experiment_sweep_execution.py`（型の検査、実行と保存、既存の結果の読込み）。runの一覧のCSVはrunのJSONの`run_metrics`から共通の処理で作られる。
- FedAvgのサーバ・記録の表・保存は`FedavgClient`に直接依存するので、再利用せず、同じ形の別のモジュールにする。

## 設計の判断
- FedAvgのclientを継承・再利用しない: 集約の配布（`apply_global_parameters`）と専門家1つを前提にしていて、DWMの専門家の集合と合わない。部品（分類器・optimizer・更新）を同じ順で呼ぶことで学習器を同じにし、一致testで確かめる。
- 新しい専門家は事前学習の直後の状態の複製: 論文の「新しい学習器」に当たる。FedHASPの新しいモデルの出発点（事前学習の初期モデル）とも、`fedavg`の出発点とも同じ。
- 規則を純粋な関数に分ける: 図1の順序（減衰→投票し直し→正規化→削除→追加）を、学習・乱数から切り離して手計算の例で確かめるため。
