# Design Document

## Overview
新実装の現在の動作をbaseline_version=1として固定する。これは正しさを初めて証明するoracleではなく、完了済み移植の実旧対照・既存goldenと独立レビューを土台に後続差分を検出する基準である。数値計算を変更しない。

## Boundary Commitments
### This Spec Owns
test側のケース定義・新APIからのsnapshot・独立process生成・環境選択・比較・新goldenとCIへの接続。
### Out of Boundary
src、旧実装、旧golden、旧回帰test、tools、LEGACY/IMPROVEの採用、依存更新、実験保存APIの変更、全構成×全dataset×全seedの直積。
### Allowed Dependencies
標準ライブラリと既存NumPy/PyTorch、srcの公開設定・実行・記録表・設定直列化・パラメータsnapshot APIだけをtest側から読む。旧testからhelperをimportせず、式や学習操作を複製しない。
### Revalidation Triggers
run設定、指標、記録列、実行順、パラメータ、依存環境の変更。意図した動作変更は後続specで差分理由を説明し基準版を追加する。通常のcommitごとには版を増やさない。形式だけの変更はschema_versionを変更する。

## Architecture
既存の13構成builder → 完全設定の明示的な代表条件へのreplace → 各手法の公開execute → 公開record tableとmetrics/最終parameter snapshot → test側のcanonical JSON → 環境別golden比較。
生産コードからgoldenへの依存を持たない。baseline版・schema版はいずれも整数1、現在の版は定数で明示し最新directoryの自動選択はしない。

## File Structure Plan
### New Files
- `tests/refactoring/behavior_golden_cases.py`: 旧依存なしのケース・snapshot・環境・生成CLI。pytest自身もimportしない。
- `tests/refactoring/test_important_behavior_regression_goldens.py`: 構造・比較検出力・数値回帰・ケース経路の検証。
- `tests/refactoring/behavior_regression_goldens/v1/windows-amd64-python3.13.15-numpy2.4.6-torch2.12.1+cpu.json`: 固定Windows数値基準（環境system-machine-python版-numpy版-torch版を小文字にして生成する命名規則）。
- `tests/refactoring/behavior_regression_goldens/v1/linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json`: WSL数値基準。
### Modified Files
- `pytest.ini`: numerical_golden markerの登録。
- `.github/workflows/python-quality.yml`: 新数値goldenだけをmarkerで必須testから外し、構造testを残す。診断stepに新数値回帰を含める。
- `docs/experiments/refactoring-baseline.md`: 新基準の生成/照合、2種類の版、環境差、意図した変更の扱い。
- `.kiro/steering/resume.md`: 現在地のみ。
共用freshは既に3手法の実runを通り、このspecは新しい生産経路を追加しないので変更せず直接実行する。

## Components and Interfaces
### ケースとsnapshot
既存13構成（FedSDA 11構成、FedDrift、FedAvg各1）のsine2小run: seed37/client2/標本35/同期10、概念gap1/probability0.5、事前学習24標本×2epoch/batch8、隠れ幅5,4。FedSDAはlocal batch4と、契約で同値を要求するcandidate maximum batch4へ変更、方式はbuilderの選択を維持。FedDriftはround2/evaluation batch3、FedAvgはlocal interval3/updates2/batch4。13ケースは構成選択と基本学習の基準で、各方式の全イベントを通ると主張しない。
追加4ケース: FedSDA sine2（seed0/client3/1500件/同期50、事前学習100×10epoch、gap100/prob0.015）、FedSDA mnist2（seed0/client2/100件/同期50、事前学習100×2epoch、gap30/prob0.015、lr0.01）。両者は旧最終構成goldenに対応する既存の完全条件を新APIで組み立てる。FedDrift sine2（16件を処理し生成末尾1件を処理しない。終端操作の成功を記録。seed71/client2/17件/同期4、gap1/prob1.0、検出閾値0.03、round3）とFedAvg（seed37/client2/9件/同期10）を追加。経路検査はケースごとに行う。sine2でcandidate_acceptedのtrue/false両方、first_segment_margin_failed、alternative_reference_within_historical_loss_tolerance、training_model_switch、model_clustering.merged_with_other_modelsのtrue、fixed_share再較正replayed標本数>0を必須とする。mnist2でcandidate_acceptedのtrue/false両方、second_segment_margin_failed、training_model_switch、fixed_share再較正replayed標本数>0を必須とする。ケース間で不足を補わない。
metricsは公開serializerのnon_finite_float_as_none=Trueで全field保存。設定も同serializerで全field保存する。各記録列と最終global parameterはshape/dtype/sha256を保存する。digestはdtype.name/shapeとtolist値をcanonical JSONへまとめ、NaN/+Inf/-Infを別のtagへ正規化、keyをsort、allow_nan=False、UTF8とSHA256を使う。NPZ ZIPのtimestampや圧縮bytesをhashに含めない。parameterはモデルID・名前をkeyにし共有部も含める。数値は同一環境で完全一致。指標の非finite値は公開serializerに従いnullへ統一し、記録列のNaN/+Inf/-Infは区別する。record列集合の追加/削除も検出する。
対象MNIST2の学習用画像/ラベル2ファイルSHA256を保存し照合する。元データをGitへ含めず、既存data readerの指定先を読む。
coverageはケースごとにイベント件数・理由集合・同期/生成/処理数の小さい要約としてsnapshotへ保存し、最低限の必須経路を別のassertで確認する。

### 生成CLI
`python tests/refactoring/behavior_golden_cases.py --output <新規json> --source-commit <40桁commit>`。
source commitは現在のHEADと同じであり、生成開始時のsrc差分が空であることを確認する（test/docの作業差分は許容）。OMP/MKL=1、torch CPU/default float32/thread1を確認する。親processは2つの新processを逐次実行して全ケースsnapshotを取得し、全値一致・必須経路確認後に出力をexclusive createする。不一致/失敗なら指定出力を作らない。中間ファイルはTemporaryDirectoryで管理。通常pytestに生成操作はない。

### 読込みと比較
基準版・形式版・環境・commit・ケース集合/設定・snapshot形を検証。envはsystem/machine/python/numpy/torch/device/dtype/threads。一致候補が0なら数値testのみskip、同一環境の重複や壊れたJSONはエラー。2環境の定義は同一で数値だけ別。比較は指標のfield path、記録列名、モデルIDとparameter名、設定のfield pathまでを示す。digest列の元の異なる要素値・位置までの表示は今回の要件に含めない。調査時は同条件runの公開記録表を読み、原因を特定する。内容のmetric/column hash/parameter hash/settingsを一つずつ変えたコピーで検出力を確認し、元goldenやsrcを改変しない。

## Requirements Traceability
| Requirements | Components |
| --- | --- |
| 1.1, 1.2, 1.3 | ケース、snapshot、経路assert、数値test |
| 1.4, 2.4 | 基準loader、項目比較、破損/差分test |
| 2.1 | 基準JSON、版directory、来歴 |
| 2.2, 2.3 | 生成CLI、独立process、exclusive output |
| 3.1, 3.2 | 2環境golden、env selector、常時構造test |
| 3.3 | marker、CI必須/診断step |
| 3.4 | source-only helper、import境界test、共用fresh |

## Testing Strategy
TDDでsnapshot不一致・未知版・欠損列・重複環境・生成の既存先拒否を先に検査する。全ケースは2process生成の一致と環境別数値回帰をWindows/WSLで確認する。markerのcollectionを読み、必須testに構造testが残ることを確認する。独立CLIレビュー指摘反映後に全pytest、Ruff、Pyright、pip check、依存境界、共用fresh、固定旧差分なしを確認する。旧回帰の2goldenや既存環境goldenを上書きしない。変異testは既定の省略。
