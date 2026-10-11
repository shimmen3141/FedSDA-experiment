# Requirements Document

## Project Description (Input)

ablationの構成の、移植の7つ目と8つ目。提案構成の、損失の監視の検出器を、e-SRから、ADWINへ替えた2つの構成を、新実装で実行できるようにする。2つは、同じ検出器と、同じ補助の判定を使うので、1つのspecにまとめる。

旧実装での指定（`tools/run_main_ablation_suite.sh`）:

- `class-adwin`: 構成名`FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting`、`--fixed-adwin-delta 0.05`。client（`ClassConditionalADWINFedSDAClient`）は、全体の損失の窓と、正解クラス別の損失の窓を、ADWIN（`FullScanADWIN`）で監視する。どれかが検知したら、警報。
- `overall-adwin`: 構成名`FedSDA_NoCached_ResidualAdapter_ADWIN_RestartingSoftRouting`、同じδ。client（`ADWINFedSDAClient`）は、全体の窓だけ。

どちらのclientも、検出器が検知しなくても、補助の判定（`_forced_drift_check`。既定で有効）で、警報を出す: 全体の窓の標本数が、保留の容量の近くのとき、窓にあたる直近の保留標本で、現在のモデルの平均損失が、履歴の平均より、距離の閾値以上に大きければ、警報。e-SRのclientは、この判定を使わない。ADWINのclientは、監視の値（e値）を持たない。

## Requirements

### Requirement 1: ADWINの検出器

#### Acceptance Criteria

1. The ADWINの検出器 shall 損失1つごとに、窓・損失の和・二乗和・検知・調べた分割点の数が、旧の`FullScanADWIN`と一致する（窓の上限での切捨てと、検知での、古い側の削除を含む）。

### Requirement 2: ADWINでの監視と、補助の判定

#### Acceptance Criteria

1. Where 検出器がADWINのとき, the 損失の監視 shall 全体の窓（と、監視の対象がクラス別を含むなら、正解クラスの窓）を更新し、どれかが検知したら警報とする。変化の候補の長さは、全体の窓が検知したか、クラス別の窓が検知していなければ、全体の窓の標本数。クラス別の窓だけが検知したときは、そのクラスの窓に残った、最初の標本の位置からの長さ。
2. Where 検出器がADWINのとき, the 標本1件の処理 shall 検出器が検知しなくても、補助の判定が真なら、警報の処理を行う。判定の条件と式は、旧の`_forced_drift_check`と同じ。
3. The 損失の監視 shall ADWINでは、監視の値（e値）を、負の無限大とする。
4. The run設定 shall ADWINの検出器に対応する、検出器の表示名（`ADWIN`）を要求する。

### Requirement 3: 構成

#### Acceptance Criteria

1. The 構成 shall 掃引のコマンドで、`fedsda-classwise-adwin-detection`・`fedsda-overall-loss-adwin-detection`という名前で選べる。run設定は、提案構成と、検出器（と、監視の対象）と、検出器の表示名だけが違う。
2. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、実旧が持つ指標と離散列の全部が一致する。
3. The 照合 shall 固有の経路を通る: ADWINの検知による警報、補助の判定による警報、補助の判定が偽の標本。クラス別の窓は、クラス別の構成だけが持つ。結果が、提案構成と違う条件がある。クラス別の窓だけが検知する経路は、監視の単体のtestで、旧のclientと照合する。
4. The 提案構成 shall 結果が変わらない。
