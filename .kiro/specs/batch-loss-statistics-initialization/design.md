# 設計: batch損失から初期統計を作る
## Boundary Commitments
### This Spec Owns
外部計算済みの非空標本別損失・ラベルから全体/各class初期統計を作るpure数値処理。
### Out of Boundary
損失生成、forward/学習/model実体、登録/ID/送信、store所有、merge、監視方針・警報後進行。
### Allowed Dependencies
stdlib、torchとその公開API、exact bounded_loss_moments module/BoundedLossMoments、exact model_and_class_loss_statistics module/ModelAndClassLossStatistics。
store/accumulate/推定/private helper・別module/methods/runtime/旧/NumPyは参照しない。
### Revalidation Triggers
singletonfallback、演算順、dtype/shape/class順変更は旧oracle/store/baseline接続を再検証。

## Architecture
中立learning/loss_statisticsのpure関数として既存immutable型を返す。入力損失生成は上位のモデル部品の責務。
既存平均損失APIへper-sample生成を追加せず、外部計算済みvectorを受ける。
NumPy/逐次Welfordでbatch統計を再計算すると数値順が変わるためtorch reductionを直接採用。

## File Structure Plan
- src/federated_learning_experiments/learning/loss_statistics/batch_loss_statistics_initialization.py: 入力検査と初期集計。
- tests/refactoring/test_batch_loss_statistics_initialization.py: 旧register oracle/異常/独立/副作用/明示接続。
- tests/refactoring/test_single_run_dependency_boundaries.py: exact数値依存許可と禁止注入。
- 対象spec/roadmapと、実際の新発見があれば既存台帳。

## Components and Interfaces
initialize_model_and_class_loss_statistics_from_batch(*,per_sample_bounded_losses:torch.Tensor,observed_class_labels:torch.Tensor,class_count:int)->ModelAndClassLossStatistics。
_validate_batch_loss_statistics_inputs: 同じkeyword入力、返却None、全検査をreduce前に完了。
class_count exactbuiltinint>=2。各tensorはCPUfloat32/dense strided。損失shape[N]、labels[N,1]、N>0/一致。
loss有限0〜1、label有限/整数値/0<=label<class_count。非連続strided入力も許可。device meta/GPU、sparse、float64/int/boolは拒否。
torch.no_grad境界でinput非変更、返却はPythonscalarの既存frozen型。共有RNG/default dtype/device/呼出元gradmodeを保存する。
全体mean torch.mean.item→float、N>=2variance torch.var(correction=1).item→float、M2=variance*max(1,N-1)をPythonfloatで計算。
N=1はvariance0.1の明示分岐（旧NaN fallbackと同じ数値、未定義分散warningは生成しない）。
class_idはrange(class_count)昇順、flatlabels一致maskによる選択で入力順保持、empty省略、mean同順、classN>=2variance同順、N1variance0/M2=0。
集計を逐次Welfordに変換せず、全fieldをBoundedLossMoments/ModelAndClassLossStatisticsのpublicconstructorで検査する。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1,1.2,1.3 | torchbatch集計、昇順class/欠落、全体n1.1/class0、旧oracle |
| 2.1,2.2,2.3 | reduce前全検査、scalarfrozen、input非変更/RNG/default/gradmode |
| 3.1,3.2,3.3 | 旧registerstub oracle、seed→store→record→baseline、exactAST/fullgolden/CPU独立smoke/正本 |

## Error Handling
TypeError/ValueErrorで項目名と理由を示す。入力/外部状態を変更しない。singletonは明示旧数値方針で、他のNaNをfallbackで隠さない。

## Testing Strategy
task1 TDDで旧BaseClient._register_trained_new_modelを学習しないstubから直接呼び出し、binary/multiclass/allzero/allone/noncontiguous/classmissing/onebatch/classsingleton/入力順/数値丸めの全fieldを完全一致比較。
task2 test-onlyでwrongdtype/device/layout/shape/empty/mismatch/range/finiteness/fractionlabel/boolclasscount、入力変更/結果独立frozen/grad/RNG/default、storeへ明示保存/更新と既存baseline選択を検証。
task3 exactAST許可/禁止注入、全tests旧11/最終3golden、CPUtorch freshsmoke、9条件/roadmap/発見事項とLunaGO。
