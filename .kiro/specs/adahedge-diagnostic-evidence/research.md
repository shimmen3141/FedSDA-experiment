# 調査

## 参照と最小境界

旧expert_routing.pyのAdaHedgeRouter（141–381行）のうち、初期証拠、集合同期、learning_rate、probabilities、update、restart_for_conceptに対応する。集約関連の計数・replay・leader選択・effective_expert_countは本specへ含めない。

旧clients/fedsda.py 2052–2062行の通知はglobal、生成済みcontext、shadow-metaのAdaHedgeを再始動し、Fixed-Shareとoracle概念別AdaHedgeを再始動しない。最終構成ではcontext/meta状態は空（1687–1711行の別方式に限定）。globalは1609–1614、1651–1653、1830–1847行の診断、oracle概念別は1663–1672、2037–2039行に使われる。真の概念入力は状態選択の後続責務で、本ownerの入力に導入しない。

新srcにはFixed-ShareはあるがAdaHedge証拠ownerはない。通知だけの先取りProtocolを作る案を避け、単一ownerを前提部品として切り出す。保存再始動回数/global gain/条件付きLOOは最終3goldenで比較されない。旧の通知接続を後で実旧clientと照合する必要がある。

## 手順と検証計画

fable-method、kiro-spec-init、kiro-spec-requirements、kiro-spec-design、kiro-spec-tasksを参照。要求のEARS・数値ID・範囲・隣接責務、設計の境界・要求対応、逐次taskの完了条件を確認した。

2026-10-08、Windows基準venvで実旧`AdaHedgeRouter`の取得→更新→再始動を直接実行し成功（exit0）。ID [-1,2]の重みは各0.5、損失0.2/0.8の更新後gap=0.3、再始動後空/計数1。式のtest内複製をoracleにしない。module自体はstdlibだが旧packageのimportでtorch/matplotlibも読み込まれるため、基準のMPLCONFIGDIRを指定する。最初の試行はmatplotlib既定cacheの権限警告があったが、基準cache指定後は警告なしで成功。worktreeのtestは仕様段階では未実行。

一般化は単一ownerの複数生成に留め、contextや通知の抽象を先取りしない。追加ライブラリは採用せず、旧数値計算の順序を保持する。既存のFixed-Share ownerを流用しない。数値・入力拒否・copy契約を変えたときの後続再検証をdesignへ明示した。

研究アルゴリズム・診断削除・改善候補の採用は行わない。WSLの既知3失敗を理由にgoldenを変えない。Linux用goldenは新全体run接続前の別spec。

## Task 1の実測と手順上の事実

Windows基準でowner実装前RED（未作成moduleでcollection1error）、注入guard追加前RED（10 failed/12 passed）を確認。実装後、対象＋依存境界2860 passed、Ruff check/format、基準Python指定のPyright成功。全回帰・変異・fresh単独動作はこの時点で未実施。

最初のguard追加で既存param tupleへ文字列が誤挿入されcollection失敗。局所修正後、独立debuggerが既存静的param1776件のarity異常なしと新22guardの実関数評価一致を確認し、最後に全対象＋境界suiteが成功。最初のPyrightは基準Python未指定で既存数値依存を解決できず、指定して成功。

一時的に作った未登録test名は登録済み正常列testへ統合し、新しい公開名を残さなかった。test専用Mappingは命名r2承認を待ってから追加したが、説明した標準dunder3メソッドの表登録が漏れ、namesが検出した。r3で明示して独立レビューへ戻した。後続実装は登録名の機械照合をレビュー前に継続する。

通常sandboxでPythonの作成直後のtemp directory内への書込み/削除が拒否された。通常のrequire_escalatedによる同じ短いprobeで成功し、SAC設定やvenvは変更していない。全回帰は同じ承認経路のWindows基準環境で実施する。
