# Requirements Document

## Introduction
FedSDAのclientで、警報の推定変化区間が最小件数（`minimum_change_interval_sample_count`。最終構成は5）より短いと、モデルの再利用も候補検証も行わずに、検出器だけを再開する（短い警報）。検出器は、数標本で、また短い警報を出し、これが続く（連鎖）。連鎖の間、新しい概念の標本は、現行の（古い概念の）モデルへ取り込まれ続ける（[IMPROVE-006](../../../docs/research/improvement-candidates/improve-006-keep-detection-evidence-after-too-short-alarm.md)。2026-10-13の観測: sine2で、1連鎖が平均237標本。試行では、連鎖が無くなり、変化から切替までの遅れが約34標本から5〜13標本へ縮み、精度が+0.3〜+0.9ポイント）。

この問題への対処として、「推定変化区間が最小件数に達するまで、警報を出さずに待つ」扱いを、設定で選べる形で足す（2026-10-13、ユーザーが、案と進め方を了承）。**既定は変えない**（いまの扱いのまま）。採否は、研究室サーバでの、複数seedの比較の後に、ユーザーが決める。

## Boundary Context
- 対象: 新FedSDAのclientの、標本1件の処理の、警報を出す条件。その設定。比較のための実行script。
- 対象外: ADWINで監視する構成での待つ扱い（下の要求1.4。ADWINは、検知したときに窓の古い側を削るので、再開せずに待っても、合図が続かない）、既定の変更（このspecでは行わない）、検出器の中身（e値、ADWIN）、最小件数の値、短い警報の処理そのもの（いまの扱いを選んだときの動作は、変えない）、FedDrift・FedAvg、固定旧と旧golden、重要動作のgoldenの新しい基準版（既定が変わらないので、足さない）。
- 用語: 「検出器の合図」は、損失の監視（e-SR、または、ADWINと、ADWINの補助の判定）が、変化ありと判定したこと。「警報」は、合図を受けて、標本1件の処理が、警報の処理（警報の記録、区間の準備、再利用・候補検証、監視の再開）を行うこと。「推定変化区間の件数」は、警報の処理が変化区間とする標本の数（保留の標本の数と、検出器の推定の長さの、小さい方）。

## Requirements
### Requirement 1: 変化区間が最小件数に達するまで、警報を待つ
**Objective:** As a 研究者, I want 短い警報の連鎖が起きないこと, so that 変化の後、新しい概念の標本が古いモデルへ取り込まれ続けず、適応の判断が早く始まる。
#### Acceptance Criteria
1. Where 待つ扱いを選んだ, when 検出器の合図があり、推定変化区間の件数が最小件数より少ない, the FedSDA system shall その標本を、警報の無い標本として処理する（警報を記録しない。区間の準備・再利用・候補検証を行わない。検出器を再開しない。保留を消費しない。保留の容量を超えた最古の標本の確定と、学習要求の記録は、警報の無い標本と同じに行う）。
2. Where 待つ扱いを選んだ, when 検出器の合図があり、推定変化区間の件数が最小件数以上, the FedSDA system shall いまと同じ警報の処理を行う。
3. Where 待つ扱いを選んだ, the FedSDA system shall 短い警報（適応の結果`alarm_change_interval_too_short`）を、1度も記録しない。
4. The FedSDA system shall 待つ扱いを、検出器がe-SRのときにだけ受け付け、ADWINとの組合せは、設定の検査で拒否する（ADWINの構成は、いまの扱いのまま）。
5. Where 待つ扱いを選んだ, the FedSDA system shall 候補検証を保持している間の合図にも、同じ条件を適用する（推定変化区間の件数が最小件数以上になってから、検証中の警報として処理する）。
6. The FedSDA system shall 待つ間も、検出器への損失の観測、予測と予測の重みの更新、候補検証の進行を、警報の無い標本と同じに続ける。
7. Where 待つ扱いを選んだ, when 検出器の合図があり、推定変化区間の件数が最小件数に達した, the FedSDA system shall その最初の標本で、警報とする（証拠が最大の候補の開始点が変わらない間は、推定の長さが標本ごとに1ずつ伸びるので、最初の合図から警報までの標本数は、最小件数より少ない。証拠が最大の候補が、より新しい開始点へ替わると、推定の長さが縮み、警報まで、それより長くかかることがある。待つ間に、合図が止んだ場合は、警報としない）。

### Requirement 2: 設定で選べ、既定は変えない
**Objective:** As a 研究者, I want いまの扱いと待つ扱いを、同じコードで比べられること, so that 採否を、同じ条件の実験で決められる。
#### Acceptance Criteria
1. The FedSDA system shall この扱いを、検出の設定（`LossChangeDetectionSettings`）の1つの値として持ち、保存するrun設定へ含め、宣言した値以外を拒否する。
2. The FedSDA system shall 最終構成と、その派生の構成の既定を、いまの扱い（合図があれば、すぐに警報とする）のままにする。
3. Where いまの扱いを選んだ, the FedSDA system shall このspecの前と、挙動（全部の状態・指標・記録の表・乱数の消費）が同じ結果にする。保存するrun設定は、新しい項目が増え、runの識別子（設定から計算する）は、FedSDAの全runで変わる。
4. The FedSDA system shall 通常の実験のコマンド（`run_experiment_sweep.py`）の引数には、この扱いを出さない。比較用のscript（`run_behavior_fix_comparison.py`）からは、選べるようにする。

### Requirement 3: 既存の基準と照合を保つ
**Objective:** As a 開発者, I want 既存の照合が、値を変えずに通り続けること, so that この追加が、既定の動作を変えていないことを確かめられる。
#### Acceptance Criteria
1. The validation shall 実旧との照合testと、重要動作のgoldenの基準版1〜3を、値とファイルを変えずに、通し続ける（保存してある設定は、新しい項目を持たないので、値がいまの扱いであることを確かめてから、除いて比べる）。
2. The validation shall 実旧との照合の入口で、新のclientが、いまの扱い（旧と同じ）であることを確かめる。

### Requirement 4: 待つ扱いの動作を確かめ、比較の準備をする
**Objective:** As a 研究者, I want 待つ扱いが意図どおりに動くことと、比較の手順, so that サーバで、いまの扱いと比べられる。
#### Acceptance Criteria
1. The validation shall 短い警報が起きる条件（`fedsda-sine2-events`の設定）の全体runで、待つ扱いでは、短い警報が0、警報のたびに推定変化区間の件数が最小件数以上、同じ標本の2回目の取込みが0、登録されないままの採用が0であることと、いまの扱いでは、同じ条件で、短い警報が1回以上あることを確かめる。
2. The validation shall 同じ条件の全体runで、待つ扱いが、変化を取り逃がしていないことを、いまの扱いとの比較で確かめる（学習帰属の切替を検出とみなした再現率が、いまの扱い以上。変化から切替までの遅れの平均——指標`change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count`——が、いまの扱い以下。再現率は、`training_model_switch_detection_metrics.detection_recall`。この条件での実測の関係を固定するもので、一般の保証ではない。成り立たなければ、要求を緩めず、記録して、ユーザーへ報告する）。
3. The validation shall 比較用のscriptへ、この扱いを替える指定と、集計を足し、小さい条件で、実行から集計まで動くことを確かめる。いまの扱いの組合せは、このspecの前に実行した結果（同じ挙動。識別子は違う）を、そのまま、集計の比較の相手に使えるようにする（集計は、dataset・集約間隔・seedで対にする。待つ扱いの組合せだけを、後から実行して足せる）。
4. The report shall 手元の小さい比較（試行と同じ9条件）の結果を、記録して、ユーザーへ伝える。
