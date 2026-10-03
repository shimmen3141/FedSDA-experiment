# 統合検証の証拠

## 対象と範囲

Fixed-Shareのモデル集合、予測前重み、損失更新、最大重み選択、損失列再生と集約後の再較正を検証した。
旧oracleをテスト側で直接呼び、重み・累積分散・4計数を丸めず比較する。新srcは旧実装をimportしない。
モデル出力のtensor混合、ClassESR、候補・FIFO帰属、学習・サーバ、最終FedSDAの新run接続は未移植。
最終承認とGO判定はspec.jsonとreview.mdが正本。

## 2026-10-03 主担当の実行結果

| 確認 | 結果 |
|---|---|
| task 4 RED（境界例追加前後の検査） | 2 failed / 54 passed / exit 1。設定依存の未許可を検出 |
| 新src全走査・全refactoring | 1887 passed / 5.22s / exit 0 |
| 全tests（schema・旧11golden・最終3goldenを含む） | 2172 passed / 3 skipped / 118.35s / exit 0 |
| 公開APIの独立プロセスsmoke | PASS / exit 0 |
| `git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json` | 空、旧productionとgolden変更なし |
| `git diff --check` | exit 0 |
| 新controller・対象test・境界checkerのTODO/秘密値検索 | 一致なし（rg exit 1） |

3 skipは既存のWindows/POSIX条件によるもの:

- test_server_sweep_wrapper_resolves_runtime_options
- test_server_sweep_wrapper_rejects_owned_options
- test_main_ablation_suite_lists_individual_variants

### 全testsのコマンド

Python 3.13.15、NumPy 2.4.6、torch 2.12.1+cpu、pytest 9.1.1、Windows CPUの既存基準環境を使用した。新規依存なし。worktreeで:

```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/fixed-share-final-20261003a
```

再検証時はbasetempに新しい未使用の検証ラベルを選ぶ。Windows Pythonのtempfile権限により、全testsはsandbox外で実行した。対象・全refactoringは共有TMP/MPLCONFIGDIRを指定してsandbox内で通る。

### Smoke

PYTHONPATH=srcで、テストfixtureや旧実装をimportせず新公開APIを呼んだ。
明示設定はalways、fixed_share_weighted_prediction、時間尺度30。3モデルの予測前重み→損失更新→2標本の集約再生→明示reset→単一負IDの重み1を確認した。
累積分散が正、集約再生回数1・標本数2、reset後回数2・重み空、単一重み1をassertし、`fixed-share public lifecycle smoke: PASS`を出力した。

## 統合評価

- task 1の独立copy出力がtask 2の予測時重み入力へ渡る。task 3は全列検査後に同じ取得・更新を順に呼ぶ。
- 全状態を一つのcontrollerが所有し、返却dictへ変更を加えても内部へ影響しない。
- 状態表・API・private補助は承認命名revision 1に対応する。
- 境界checkerはexact controller pathから同機能設定moduleだけを許す。torch・NumPy・旧module・runtime・他機能は禁止例で確認した。設定側のcore制約は保持する。
- 全22要件はdesignのtraceabilityと完了タスクの検証で覆う。現在の部品境界内に未接続処理や全体run完成を代用するstubはない。
- 固定goldenの保持は旧経路の回帰安全性、新oracle照合は新部品の同値性の証拠。最終新runの同値性は後続統合で別に検証する。
