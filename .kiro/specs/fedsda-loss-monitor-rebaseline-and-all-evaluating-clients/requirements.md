# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を設計・評価している研究者（ユーザー）。
- 今の状況（1）: clientの損失の監視（`OverallAndTrueClassLossMonitor`）の全体の検出器（e-SR）の基準は、警報の応答の`reset`でだけ設定され、次の警報まで変わらない。
  新しいモデルを採用した警報で、若いモデルの高い損失の平均が基準に固定されると、モデルの損失の水準が下がった後の概念の変化を見逃し、そのモデルが別の概念の標本を学習する
  （学習標本の混入）。混入は、違う概念の統合の前提になる（[LEGACY-023](../../../docs/research/implementation-findings/legacy-023-loss-monitor-baseline-frozen-at-alarm-reset.md)、
  [LEGACY-021](../../../docs/research/implementation-findings/legacy-021-classwise-clustering-merges-with-little-evidence.md)）。
- 今の状況（2）: サーバのクロス評価は、対象のモデルを評価させるclientを、標本を返せるclientから最大3つ無作為に選ぶ。今その概念にいるclientが評価から外れると、
  違う概念の統合が起きる（uraniumで統合のroundだけ上限を外す対照で、5対のうち4対の統合が起きなかった。LEGACY-021の追記）。
- 何を変えるか: (1) 損失の低下を検出したら監視の基準を下げる扱い（[IMPROVE-031](../../../docs/research/improvement-candidates/improve-031-loss-monitor-rebaseline-on-loss-decrease.md)の案A）を、
  製品の正式な設定として足す。(2) クロス評価で1つのモデルを評価させるclientの上限を、clientの数にする。(1)(2)を既定にし、基準版13を足す。

## Introduction
2026-10-27、研究ツールの変種（研究用の入口`tools/research_loss_monitor_rebaseline.py`）の本規模（uranium、6データ×間隔4つ×seed 0〜9、MNIST4はseed 0〜19、各client 5000標本）で、
案Aは実験の前に固定した採否の基準（精度・仕組み・費用）を満たし、案Aと上限の撤廃を合わせた構成（`product-default-all-evaluating-clients-loss-monitor-decrease-rebaseline`）も同じ基準を満たした
（精度の差は間隔ごとの6データの平均で+0.096〜+0.126pp、データ別の最小はSEA2の−0.019pp、SEA以外で変化の見逃しと学習標本の混入がほぼ無くなる、通信量の比1.003〜1.015、MACの比1.025）。
ユーザーは、案Aを上限の撤廃とともに基準版13に入れると決めた（2026-10-27「上限の撤廃とともにv13に入れたいです」、組合せの結果の後に「それで進めてください」）。
testは、重要なもの・後で効くもの・複雑なものに絞る（基準版12のspecと同じユーザー指示）。

設定の値と呼び方の対応:

| 設定 | 値 | この文書での呼び方 |
|---|---|---|
| `LossChangeDetectionSettings.loss_monitoring_baseline_update_policy`（新しい項目） | `keep_baseline_until_alarm` | 「基準を警報まで保つ扱い」（今の扱い。基準版12まで） |
| 同上 | `lower_baseline_on_loss_decrease` | 「損失の低下で基準を下げる扱い」（新しい既定。基準版13） |
| `FedsdaRunParticipantSettings.maximum_evaluating_client_count_per_model`（既存の項目） | 3 | 「上限3」（基準版12まで） |
| 同上 | 実験のclientの数 | 「上限なし」（新しい既定。基準版13） |

用語:
- **固定された基準**: 全体の検出器（e-SR）が実際に使っている基準の平均（`reset`またはコンストラクタで与えられ、範囲[10⁻⁶, 1 − 10⁻⁶]へ制限された後の値）。
- **今のモデルの保存平均**: 標本ごとに監視へ渡す`current_model_baseline_loss_mean`（`select_loss_monitoring_baseline_mean_loss`の値。範囲[0.01, 1 − 10⁻⁶]）。
- **下がる向きの検出器**: 製品の`BoundedLossESRDetector`を、観測1 − ℓ、基準1 − （固定された基準）で動かす検出器。増分は 1 + λ(μ − ℓ)/(1 − μ)で、帰無仮説「損失の平均 ≥ μ」の下で期待値1以下、非負（増分は1 − λ以上）。
  誤警報の水準・賭け率・候補の上限は、全体の検出器と同じ。

## Boundary Context
- 対象: 検出の設定の新しい項目と検査、監視器の下がる向きの検出器と基準の引下げ、監視の計算回数への算入、基準の引下げの回数の記録と指標、最終構成の既定（2つ）、
  構成の扱い（候補検証ありの構成とADWINで監視する構成）、旧と比べる設定の入口、基準版のgolden（版1〜12の明示と、版13の追加）、既定に依るtest、研究ツールの基点の固定と意味の保持、文書。
- 対象外: 上がる向きの検出器の判定と閾値、警報の応答、警報の後の判定（最小記述長）、帰属・保留、クロス評価の評価そのもの・候補の選び方（版12の扱い）、クラスタリングの判定・統合、
  予測の混合、通信量の数え方、FedDrift・FedAvg、固定旧と旧golden、IMPROVE-031の比の閾値の変種の製品化、案B（同期の時点の再設定）、再設定の回数を減らす案。

## Requirements
### Requirement 1: 損失の低下で監視の基準を下げる扱いを選べる
**Objective:** As a 研究者, I want 若いモデルの高い基準が固定されたままにならない監視, so that 新しいモデルの学習中でも、次の概念の変化を見逃さず、学習標本の混入が起きにくい。
#### Acceptance Criteria
1. The FedSDA system shall 検出の設定に、監視の基準の扱いを選ぶ項目を持つ（値は「基準を警報まで保つ扱い」と「損失の低下で基準を下げる扱い」の2つ）。「損失の低下で基準を下げる扱い」は、検出器がe-SRのときだけ選べる（ADWINなら設定の誤り）。
2. When 「損失の低下で基準を下げる扱い」が選ばれている, the FedSDA system shall 監視器ごとに、下がる向きの検出器を持ち、固定された基準が設定されるたび（作成時と`reset`のたび）に、新しい固定された基準で作り直す。
3. When 「損失の低下で基準を下げる扱い」が選ばれている, the FedSDA system shall 標本ごとに、今の監視（上がる向きの全体とクラス別の検出器）の観測を今と同じに求めた後、下がる向きの検出器に1 − （観測の損失）を観測させる。
   今の監視の観測が警報の合図（`drift_detected`。全体とクラス別の混合の判定）なら、基準を下げない。合図がなく、下がる向きの検出器が合図を出し、今のモデルの保存平均が固定された基準より小さければ、
   監視を今のモデルの保存平均で`reset`する（全体の検出器・クラス別の検出器・標本位置を消して作り直す。警報の応答の`reset`と同じ処理）。合図を出したが保存平均が固定された基準以上なら、下がる向きの検出器だけを今の固定された基準で作り直す。
   基準の引下げは警報ではなく、帰属・保留・適応の記録・予測の重みを変えない。その標本の観測として返す値（警報の合図、log E、変化の推定、標本の位置）は、引下げの前の今の監視の値とする。
   同じ標本の後続処理のうち、log Eの記録・警報の判定・監視の計算回数の記録は、この返す観測を使い、今と同じに動く（引下げの標本では合図が無いので、警報の処理は起きない）。
   保留・帰属・標本の位置の管理は、今と同じ入力と状態で行い、引下げの影響を受けない。
   引下げの後、監視器の直前の観測（`last_observation`）と監視器の標本位置は、警報の応答の`reset`の後と同じく消える。次の標本の始めの検査のうち、監視器の位置の連続性の検査だけが警報の応答の後と同じ扱いになり
   （直前の観測が無いので検査しない）、保留の位置の連続性の検査は今と同じに行う。
4. The FedSDA system shall 下がる向きの検出器の計算を、監視の計算回数（検出器の部品の更新の回数と、調べた候補×賭け率の数）の合計に足し、さらに、下がる向きの検出器の分だけを別の項目として記録する
   （合計から別の項目を引くと、下がる向きの検出器を持たない場合の値になる）。
5. The FedSDA system shall 基準を下げた事象を、記録の表（clientごと・標本の位置・下げる前の固定された基準・下げた後の固定された基準）と、runの指標（全clientの回数の合計）に残す（「基準を警報まで保つ扱い」では0件）。
6. When 「基準を警報まで保つ扱い」が選ばれている, the FedSDA system shall 基準版12と同じ値・同じ乱数の消費・同じ通信量・同じ計算回数で処理する（下がる向きの検出器を作らず、監視の計算回数にも足さない）。

### Requirement 2: クロス評価で評価させるclientの上限を、clientの数にする
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定で、1つのモデルを評価させるclientの上限（`maximum_evaluating_client_count_per_model`）を、実験のclientの数にする（上限を超えないので、候補の全部を評価させ、上限のための乱数を消費しない。版12の候補の選び方の規則のまま）。

### Requirement 3: 既定を変える
#### Acceptance Criteria
1. The FedSDA system shall 最終構成の既定を「損失の低下で基準を下げる扱い」と「上限なし」にし、候補検証ありの構成とADWINで監視する構成を除くFedSDAの全部の構成が引き継ぐ。
   候補検証ありの構成（`fedsda-with-candidate-validation`）は、「基準を警報まで保つ扱い」と「上限3」を明示する（候補検証との組合せは未検証のため。版12と同じ考え方）。
   ADWINで監視する構成（`fedsda-classwise-adwin-detection`、`fedsda-overall-loss-adwin-detection`）は、「基準を警報まで保つ扱い」を明示し、上限は既定（上限なし）を引き継ぐ。全部の構成の組立てが例外を出さず、実験のコマンドの引数を変えない。

### Requirement 4: これまでの基準と照合を保つ
#### Acceptance Criteria
1. The validation shall 実旧との照合test（部品・全体run・構成）を、変わらず成功させる。旧と比べる設定の入口と、実旧との照合に使う設定の組は、「基準を警報まで保つ扱い」と、その組の今の上限を明示する。
2. The validation shall 基準版1〜12のgoldenを、「基準を警報まで保つ扱い」と「上限3」で、値を変えずに照合し続ける（全部の版を、既存の版ごとの設定の組立てと照合のtestで確かめる）。

### Requirement 5: 新しい既定を基準版13として足し、研究で評価した値と同じであることを確かめる
#### Acceptance Criteria
1. The validation shall 新しい既定での値を、基準版13のgoldenとして、Windowsの基準環境とWSLの両方で足す（ケースは版12と同じ22ケース）。
2. The validation shall 版12と13の差を、両方の版を実行した結果を比べて確かめる: FedDrift・FedAvgのケースと、候補検証ありの構成のケースは、指標・最終のパラメータ・記録の表・経路の記録（版13で足す項目を取り除いたもの）が版12と同じ。
   新しい既定を引き継ぐFedSDAのケースは、設定が、新しい項目と上限だけ違う。
3. The validation shall 経路の記録に、基準を下げた回数を足し（版13から。FedSDAのケースだけ）、候補検証ありの構成のケースでは0、新しい既定を引き継ぐFedSDAのケースのうち少なくとも2つ
   （既定の候補は`fedsda-sine2-adoption-without-validation`と`fedsda-mnist4-warm-started-learner`。生成の前に主担当が値を確かめ、0なら基準を下げるケースを選び直して理由を記録する）で1以上であることを、必須の経路として確かめる。
   版12までの必須の経路は、版13でも求める。
4. The validation shall 既定を変える前に、製品の全体run（基準版12の既定に、新しい2つの値だけを置き換えた設定）と、研究用の入口の組み合わせた変種
   （`product-default-all-evaluating-clients-loss-monitor-decrease-rebaseline`、製品の掃引のraw）の、同じ条件の結果が、許容差なしで一致することを、
   複数のデータセット（MNIST2と、SINE2・CIRCLE2・SEA4。各条件で基準の引下げが1回以上起き、評価clientの候補が3を超える同期を含む）で確かめる。一致の定義:
   (a) 研究のrawにある記録の表の全配列が、製品のrawの同じ名前の配列と`numpy.array_equal`（浮動小数は`equal_nan=True`）で一致する（標本ごとのlog E・警報・判定・適応・統合・クロス評価・予測を含む）。
   (b) runの要約の全項目が一致する。除くのは、実行の時刻・秒数・出力先の名前・コードの来歴、設定の新しい項目（`loss_monitoring_baseline_update_policy`）、
       製品だけが持つ項目（基準を下げた事象の記録の表と回数の指標、監視の計算回数の下がる向きの検出器の分の項目）、監視の計算回数の合計。
   (c) 監視の計算回数: 製品の合計から下がる向きの検出器の分を引いた値が、研究の値（合計）と一致する。
   (d) 基準を下げた事象: 製品の記録の表の、clientごとの（標本の位置、下げる前の基準、下げた後の基準）の列が、研究の診断の`rebaselines`のうち同じclient IDの事象の列と一致する。
       研究の診断は、各事象にclient IDを持つ（研究ツールの変更。下の要求6.4）。回数の指標は、研究の診断の`rebaseline_count`と一致する。

### Requirement 6: 研究ツールの意味を保つ
#### Acceptance Criteria
1. The research tool shall 既定を変えた後も、`product-default`系以外の既存の変種が、基準版10の既定の上で評価したときと同じ意味を持つようにする（研究ツールの基点の固定に、新しい項目の今の扱いと上限3を足す）。
2. The research tool shall IMPROVE-031の研究用の変種（`product-default-loss-monitor-*`と、組み合わせた変種）は、既定を変えた後も、基準版12の既定の上で評価した意味を保つ
   （研究用CLIと研究用の入口の両方で、製品の新しい2つの値を基準版12の値に戻してから変種を当てる）。`product-default`は、製品の既定（基準版13）を使う。
3. The validation shall 既定を変えた後、研究用CLIの`default`と`product-default-loss-monitor-decrease-rebaseline`、研究用の入口の`product-default-all-evaluating-clients-loss-monitor-decrease-rebaseline`の小さい条件の結果が、変える前と同じことを確かめる。
4. The research tool shall IMPROVE-031の研究用の変種の診断の各事象（`rebaselines`の要素）に、その監視器のclient IDを持たせる（監視器の観測の時点で、実行中の`process_observed_sample`のclient IDを読む。
   `tools/research_loss_monitor_baseline_trace.py`と同じ方法）。診断の項目を足すだけで、runの結果を変えない（研究ツールの専用の検証の観察だけの照合で確かめる）。

### Requirement 7: 文書を合わせる
#### Acceptance Criteria
1. The documentation shall 手法の説明（損失の監視の基準、クロス評価の上限）、基準の文書（版13）、実験の入口、研究記録（LEGACY-023・LEGACY-021の状態、IMPROVE-031、v12・v13比較）、実装の不具合の一覧、再開案内で、新しい扱いを書く。
