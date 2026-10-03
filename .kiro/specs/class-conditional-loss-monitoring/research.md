# 調査と設計判断

## 旧基準の所在

- federated_drift_experiment/drift_detectors/e_detector.py: 単一BoundedMeanEDetectorのfloat64候補×賭け率capital、対数混合、候補番号・保持・reset。
- clients/fedsda.py:1126: 全体e-SR、モデル統計からbaselineを取得（統計なし0.01、通常0.01～1−1e-6）。
- clients/fedsda.py:1174: ClassESRの遅延class生成、全体/今回class/他class初出順の混合、global位置への対応。
- config.py:176: 候補保持上限1000、alpha=0.001。賭け率既定(0.05,0.1,0.2,0.4,0.8)は検出器定数。

既存調査threadを再利用し、追加調査のみ依頼した。rootはソースを読み、requirements/designの合成を行う。fable-methodとcc-sddの要求・設計gateを参照。既存実装からの限定移植のためlight discoveryを使用し、新規ライブラリは導入しない。

## 数値と操作順

capital/賭け率はNumPy float64、候補番号はint64。毎観測で時刻を1増やし、capital=0の新候補を追加して全候補へlog増分を加える。L=1+lambda*(loss/baseline−1)、log(max(L,float64 tiny))。上限超過は古い候補から捨てる。行ごとの賭け率混合はNumPy exp.sum(axis=1)、全候補混合も旧log-sum-exp順を維持する。候補数で平均しない。np.argmaxの最初の同率候補を選び、候補年齢=current−start+1。log(1/alpha)の等号で警報。

ClassESRの混合はPython sum(exp(...))。overall重み=1/(K+1)、class重み=(1−overall)/Kであり、数学的な等重みを同じ浮動値で置き換えない。未観測classは混合に出さず、その分を再配分しない。他classは最後の値を保持する。有限logだけを混合する旧規則も維持。

classが選ばれると、候補番号−保持最古番号をclass位置dequeへ対応させる。overallが選ばれる/混合警報がない場合の候補はoverallの年齢からglobal位置へ戻す。今回の入力契約は区間内global位置を連続に限定し、class位置だけは非連続を許可する。FIFOによる帰属位置の切詰めは含めない。

## 実測と重要な差

初期状態は時刻0、幅0、log-e=−inf、e=0、候補なし、仮説数0。baselineと同じ損失2件でlog-e=log(2)、同率は最古候補1。overallbaseline=0.2でclass0初出後、モデル平均を0.7へ変更してclass1を初観測するとoverall=0.2/class0=0.2/class1=0.7になる。

旧単一e-SRの不正損失は変更前に拒否する。一方旧ClassESRの不正classはoverallを更新してから拒否し、counterも部分更新される。新APIは正常系列の一致と不正入力時のatomic拒否を区別する。bool・文字列・空bet列・非整数保持上限・NaN baselineの強制変換を残さない。

## 主担当の境界判断

既存LossChangeDetectionSettingsはalpha/監視対象の唯一所有者のまま維持する。候補保持上限と非空bet tupleは新数値部品の明示constructor条件とし、今回のためだけに汎用coreへtuple schemaを追加しない。baselineは可変なモデル統計由来なので固定設定へ混ぜず、初期/resetとclass初出時の値を明示入力とする。初回は旧等重みのみ。モデル統計・履歴蓄積・学習・client操作は後続。

単一数値状態、混合/位置状態、immutable結果を同じ機能内の小さな境界にする。既存重みcontrollerやtorch依存を監視へ持ち込まない。NumPyはexact単一数値moduleでのみ許可し、mixed monitorはstdlibと同機能型・数値部品・設定に依存する。抽象factory/frameworkは不要。

## 検証接続

単一旧検出器を直接生成し、全候補capital/番号・結果を各観測後に比較する。旧ClassESRは__new__で必要stateだけ用意するテスト専用fixtureとし、モデルforward・torch RNGを伴う旧client初期化を避ける。configのalpha/上限/class数だけをpytest monkeypatchで固定し復元する。旧client実メソッドによる更新・reset・位置推定へ直接照合する。旧importはtestのみ。

全testsと旧11/最終3goldenを更新せず実行し、新public APIの独立smokeで旧module不在を確認する。
