# Research Log

## 調べたこと
- 読込み`load_single_stream_input`は、クラス数を製品の`get_dataset_definition`から取る。製品に無いデータの系列は読めない（このspecの読込みの変更の理由）。
- 比較手法の研究ツール`river_baselines.py`は、JSONの`dataset_name`・`random_seed`・`sample_count`・`array_sha256`と、NPZの`features`・`labels`（事前学習ありでは`pretraining_features`・`pretraining_labels`も）を読む。クラス数は読まない。比較の集計`single_stream_comparison.py`は、系列のディレクトリの全JSON（`export_provenance`で始まるものを除く）の`dataset_name`・`random_seed`・`array_sha256`を読む。新しい系列のJSONにこれらを持たせ、出所の記録のファイルは書かない。
- 製品の概念列（`random_concept_schedule_generation.py`）: 概念0から始め、`標本の番号−直前の変化 > 最小の間隔`の標本で`random()`を引き、確率未満なら今以外の概念の一覧から`choice`（候補1つでも呼ぶ）。`fedsda`の既定は最小の間隔300、確率0.0015（`fedsda_final_configuration_run_settings.py`）。既存の系列（5,000標本）は変化が4〜5回。
- River 0.27.0（`~/river-env`）: `STAGGER(classification_function, seed, balance_classes)`、`Agrawal(classification_function, seed, balance_classes, perturbation)`、`RandomRBF(seed_model, seed_sample, n_classes, n_features, n_centroids)`。STAGGERの特徴はsize・color・shape（0〜2の整数）。Agrawalの`perturbation`は、ラベルを決めた後に数値の特徴（salary、commission、age、hvalue、hyears、loan）を値域×摂動の幅まで乱す（ラベルの雑音ではない。MOAの既定0.05）。5,000標本のクラスの割合（seed 0）: STAGGERの関数0は1が12%、1は55%、2は67%。Agrawalの関数0〜3は1が34%・38%・54%・38%（関数4以降は7・9が1%台など偏りが大きいので使わない）。Random RBFのseed_model 0〜3は1が45〜65%。

## 判断
- 生成は単一ストリームのパッケージに置く（研究ツールの`tools/research_single_client_stream_baselines`は連合学習の側の診断も持つので、単一ストリームの枠組みを後で切り離しやすいよう、パッケージにまとめる）。
- 製品のデータセットの定義へは足さない（連合学習の側を変えない）。系列のJSONにクラス数を書き、読込みはそれを優先する。製品の定義と同じ名前なら一致を確かめる。
- 概念列は製品の関数を呼べない（概念の数をデータセットの定義から取る）ので写しを持ち、2概念・4概念で製品と同じ列になることをtestで確かめる。
- 概念の定義はseedによらず固定し（STAGGER・Agrawalは関数、Random RBFは`seed_model=i`）、標本の乱数だけをseedで変える（既存の合成データと同じ考え方）。概念ごとに生成器を分け、再来は続きから取る（再来で同じ標本を繰り返さない）。
- Agrawalは関数0〜3（Riverの番号。原論文の関数1〜4）を使う。クラスの偏りが小さく、文献でよく使われる範囲。摂動はMOAの既定0.05。
