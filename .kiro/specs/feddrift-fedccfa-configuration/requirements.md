# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedAvg、FedAvg+Window、Adaptive-FedAvg、DriftSurf、IFCA+Window、FedEM、FedDrift、Oracle。分散した実コンセプトドリフトを直接扱う最近の手法が無い。
- 2026-11-05のユーザー判断: FedCCFA（Chen et al., NeurIPS 2024、"Classifier Clustering and Feature Alignment for Federated Learning under Distributed Concept Drift"）を比較手法に足す。公式の実装は`C:\Users\yshin\Research\FedSDA-experiment\FedCCFA`（`Chen-Junbao/FedCCFA`、`647d8b0`、MIT License）、論文は同じ場所の`FedCCFA_paper/`。問題設定が異なるので、合わせ方で不当に弱くしない。公式の実装との照合（元の条件に近い小さい設定での挙動の確認を含む）と、予備の実行での1時刻のround数 $R$ の軸を含める。
- 調査: GPT-6.1-solの静的調査`results/_sol_prompts/baseline-survey-fielding-fedccfa.out`（主担当が主要な行を確認済み）。
- 何を変えるか: 手元のFedDriftの実装（`runtime/feddrift_*`）の上に、FedCCFAの4構成（学習の窓2通り×特徴の整合の開始2通り）を足す。予備の実行で構成と $R$ を、調整の実行でDBSCANの $\epsilon$ を選ぶ。既存の構成と既定は変えない。

## Boundary Context
- 対象: 新しい4構成、DBSCANの $\epsilon$ の掃引の軸、1時刻のround数の掃引の値を4構成でも使えるようにすること、2クラスのデータ用の2出力の分類層（FedCCFAの構成だけ）、記録、test（公式の式との照合と、元の条件に近い設定での挙動の確認を含む）、文書、予備・調整・本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、FedHASP（`fedsda`）、研究ツール。FedCCFAに無い仕組み（集約の間の学習、過去の分類器の保管庫など）は足さない。公式の`oracle`の分岐と`fine_tune`（評価の前の微調整、公式でもコメントアウト）は使わない。

## Requirements
### Requirement 1: FedCCFAの4構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-fedccfa-window`、`feddrift-fedccfa-window-scaled-alignment`、`feddrift-fedccfa-window200`、`feddrift-fedccfa-window200-scaled-alignment`を持つ。どれも、全clientで共有する1つのグローバルなモデル（表現部と、使わない初期の分類層）と、clientごとの個人の分類層を持つ。表現部は分類層（`classification_layer`）より前の全部、分類層は`classification_layer`の重みとバイアス（公式の`clf_keys`＝最後の2つのキーと同じ切り分け）。コンセプトの数・変化点・コンセプトのIDは使わない。
2. The system shall 初期値を全手法に共通の事前学習した初期モデルの写しにする（表現部・初期の分類層 $\phi^{(0)}$・各clientの個人の分類層の初期値）。2クラスのデータでは、FedCCFAの構成に限り、分類層を2出力（softmax）にし、事前学習の1出力の重み・バイアス $(w,b)$ から、クラス0の行を $-(w,b)/2$、クラス1の行を $(w,b)/2$ にする（初期の予測確率は1出力のsigmoidと一致する）。
3. The system shall 学習の窓を構成ごとに次のとおりにする。`-window`: その時刻に届いた $I$ 標本。`-window200`: そのclientが直近に観測した $\max(I,200)$ 標本（観測済みの標本だけ。runの初めは届いた分だけ）。
4. The system shall 各round（1時刻に $R$ 回）で、公式の`methods/FedCCFA.py`の順に、client側で (i) 窓のラベルの分布とそのエントロピー $H$ による整合の重み $H/\gamma$、(ii) 共通の初期の分類層 $\phi^{(0)}$ から、窓のクラス均衡の部分集合で分類層だけを学習した診断用の分類層、(iii) 個人の分類層の復元と、表現部を固定した個人の分類層の学習、(iv) 個人の分類層を固定した表現部の学習（教師ありの損失＋整合の重み×対照の整合の損失）、(v) 学習後の表現でのクラスごとのアンカー（平均の特徴）を求め、サーバ側で (vi) 表現部を窓の標本数の重みで平均、(vii) クラスごとに、診断用の分類層の行のMADDの距離とDBSCAN（`min_samples=1`）でclientを分け、2 client以上の組で個人の分類層のその行とアンカーをclientの等重みで平均して戻す。式と手順の詳細はdesign.md。
5. The system shall 特徴の整合を、runの最初のroundを $r=0$ とする通算のround番号 $r$（1時刻に $R$ 個）で判定し、`-window`・`-window200`では $r\ge 20$（公式`entities/FedCCFA.py:131`）、`-scaled-alignment`では $r\ge\lceil 0.1\,(N/I)\,R\rceil$（$N$ はclientごとの標本数。論文の $T_s=20$ はFedFMの200 roundでの経験値なので、その比率を保つ）のときに行う。どちらも、アンカーがまだ無いround（$r=0$）では行わない（公式で`global_protos`が空の初回と同じ）。
6. The system shall 学習の予算・optimizerを、全手法に共通の規則に合わせる。表現部の学習の更新回数はFedDriftの1モデル分（`updates_per_sample`×`detection_batch_sample_count`）、個人の分類層はその $1/5$（切り上げ。公式の`clf_epochs`:`rep_epochs`=1:5）、診断用の分類層は5回（公式の`balanced_epochs`=5、均衡の部分集合は1ミニバッチ）。optimizerは共通のAdam（AMSGrad、weight decay 0.001）で、分類層の2つの学習の学習率は共通の学習率の10倍（公式の`clf_lr`/`rep_lr`=`balanced_clf_lr`/`rep_lr`=10）。
7. The system shall 各標本を、ラベルを見る前に、そのclientの今のモデル（直前の配布の表現部＋直前のroundの後の個人の分類層）で予測する。集約の間は学習しない（既存のFedDrift系の構成と同じ）。
8. The system shall 窓に欠けるクラスを、移植のための明示的な拡張として扱う（design.mdの「欠けたクラス」）。公式は全クラスが窓にあることを前提にしており（欠けると`class_balance_sample`は空の候補から抜こうとし、エントロピーは $0\log 0$ を計算する）、この拡張は公式の実装との照合の対象外の差として記録する。未来の標本や真のコンセプトで補わない。
9. The system shall DBSCANの $\epsilon$ を数値パラメータ（既定0.1）と掃引の軸`--fedccfa-dbscan-eps`として持ち、整合の重みの $\gamma$（既定20、正の値だけを受け付ける）、対照の整合の温度（既定0.1）を数値パラメータとして持つ。公式の $\gamma=0$ のときの`lambda`の分岐（`entities/FedCCFA.py:38-42`、公式の既定`lambda: 0.0`）は持たない（$\gamma>0$ では使われない）。1時刻のround数の掃引の値`--feddrift-local-training-round-counts`を4構成でも使えるようにする。
10. The system shall 既存の全部の構成の名前・順・保存設定・runの識別子を変えない（FedEMのときと同じく、既存の全構成の設定と識別子の固定hashの照合と、基準版1〜16のgoldenの照合で確かめる）。

### Requirement 2: 記録
#### Acceptance Criteria
1. The system shall 各時刻・各round・各クラスの、clientのクラスタの番号（単独なら $-1$）と、窓のそのクラスの標本数を記録する。各時刻・各roundの、整合を行ったかと、各clientの整合の重みを記録する。通信は、表現部・個人の分類層・診断用の分類層・アンカーを分けて数える（表の名前と列はdesign.md）。

### Requirement 3: test
#### Acceptance Criteria
1. The validation shall （公式の実装との1 roundの照合）公式のコードと手元の移植を、共通の小さい固定の入力（モデルの初期値・データ・round列）で1 roundずつ動かし、診断用の分類層・個人の分類層・表現部・アンカー・集約の後の値が許容差の中で一致することを確かめる（design.mdの「公式の実装との照合」）。あわせて、MADDの距離、DBSCANによるクラスごとの組、組ごとの分類層の行とアンカーの平均、表現部の平均、整合の重み、対照の整合の損失、クラス均衡の部分集合の件数が、公式の実装（`entities/FedCCFA.py`）の抜粋と同じ値・同じ組になることを、固定の入力で確かめる。
2. The validation shall 元の条件に近い小さい設定（design.mdの「元の条件での挙動の確認」）で、ラベルが入れ替わったclientの組が、入れ替わったクラスについてクラスタに分かれ、入れ替わっていないクラスでは群をまたぐ組ができること、精度が入れ替わりの後に戻ることを、design.mdに固定した対象の時刻・比較の区間・最低の改善幅で確かめる。満たさないときは条件を緩めず、主担当へ報告する。
3. The validation shall 小さい全体runで、round内の順序、窓、予測にラベルを使わないこと、予算、欠けたクラスの扱い、2クラスの初期の予測確率、通信の計数、記録を確かめる（design.md）。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 4: 文書と実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、4構成と公式のFedCCFAとの違い（design.md）を書き、設定参照を再生成する。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、予備の実行・$\epsilon$ の調整の実行・本規模の実行（design.md）のコマンドを順にユーザーに渡し、集計を示す。表に載せる構成・$R$・$\epsilon$ の選び方はdesign.mdの規則に従い、最終の判断はユーザーが行う。
