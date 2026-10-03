# 設計: 重み付き分類予測

## Overview

既に得られたモデル出力から混合予測と観測後損失を計算する。正常な旧最終構成の数値・演算順を維持し、予測前のAPIからラベルを除く。

### Goals

確率化→重み正規化→混合→クラス判定、観測後のモデル損失→上流重み更新を明示入力で検証する。

### Non-Goals

モデルforward・optimizer・監視・候補・FIFO・diagnostics・active集合・AdaHedge/meta・保存・新全体run。

## Boundary Commitments

### This Spec Owns

出力確率化、全pool重みの正規化、混合、クラス判定、ラベル後の平均有界損失のstateless数値処理。入力検査と独立した返却値。

### Out of Boundary

重み・モデルの永続状態を持たない。モデル別出力を供給するforwardの最適化と、手法の操作順・診断は後続spec。上流controllerのprivate APIへ依存しない。

### Allowed Dependencies

新数値moduleはstdlibとtorchだけ。モデル・method設定・上流controller・runtime・旧moduleはimportしない。テストだけが上流controllerと旧static oracleを接続する。

### Revalidation Triggers

shape/dtype/device、重み正規化、確率検査、クラス閾値・同率、平均・加算順、返却所有・依存を変えた場合に本specと後続client接続を再検証する。

## Architecture

`learning/prediction`の数値部品を下位層として配置する。今後method/clientが必要な出力と重みを渡す。torch依存を手法のstdlib重み状態へ持ち込まない。5関数は一つの予測数値責務で、独立interfaceやsnapshot型は追加しない。

| 層 | 技術 | 役割 |
|---|---|---|
| 数値処理 | Python 3.13、torch 2.12.1+cpu、stdlib math | CPU float32・旧演算 |
| 状態接続の検証 | 既承認FixedSharePredictionWeightController | 前後の同じ正規化snapshot |
| oracle | 旧clientのstaticと旧SwitchingExpertRouter | テスト専用、旧src変更なし |

## File Structure Plan

- 新規`src/federated_learning_experiments/learning/prediction/__init__.py`: 説明のみ、再exportなし。
- 新規`src/federated_learning_experiments/learning/prediction/class_probability_calculations.py`: 全stateless数値処理と入力検査。
- 新規`tests/refactoring/test_class_probability_calculations.py`: 各操作の旧oracleと重み状態接続。
- 変更`tests/refactoring/test_single_run_dependency_boundaries.py`: exact新moduleにtorchだけを許可し、package・許可/禁止例を確認。
- 変更`.kiro/steering/roadmap.md`: 完成範囲と次の境界。旧src・golden・設定は変更しない。

## Requirements Traceability

| Requirement | 契約 |
|---|---|
| 1.1, 1.2, 1.3, 1.4 | 出力→確率、CPU float32と形・有限・クラス数の検査 |
| 2.1, 2.2, 2.3, 2.4 | 重み正規化と混合、昇順積・逐次加算、同集合 |
| 3.1, 3.2, 3.3, 3.4 | 有限スコアの二値閾値/多クラスargmaxと返却形 |
| 4.1, 4.2, 4.3, 4.4 | 同じモデル確率と観測ラベルからfloat32平均損失 |
| 5.1, 5.2, 5.3, 5.4 | no_grad/copy・旧oracle・上流接続・依存/部分移植 |

## Components and Interfaces

### Classification prediction calculations

Intent: 状態を持たず出力・重み・観測ラベルを数値へ変換する。External torch（P0）、Inbound将来method/client（P2）。契約: Service。

全public引数はkeyword-only。正式名・局所名はnaming.md。

| 関数 | 引数 → 返却 |
|---|---|
| convert_model_outputs_to_prediction_probabilities | model_outputs_by_model_id: Mapping[int,Tensor], class_count: int → dict[int,Tensor] |
| normalize_model_prediction_weights | prediction_weights_by_model_id: Mapping[int,float] → dict[int,float] |
| combine_model_prediction_probabilities | prediction_probabilities_by_model_id: Mapping[int,Tensor], prediction_weights_by_model_id: Mapping[int,float], class_count: int → Tensor |
| predict_class_labels_from_prediction_scores | prediction_scores: Tensor, class_count: int → Tensor |
| compute_model_mean_bounded_losses_after_label_observation | prediction_probabilities_by_model_id: Mapping[int,Tensor], observed_class_labels: Tensor, class_count: int → dict[int,float] |

#### Preconditions / Postconditions

- 要件の入出力契約をそのまま検査する。TensorはCPU float32 strided、N>0、binary[N,1]/multi[N,K]。class_countはbuiltin int、bool除外、≥2。同一mappingの全N一致。
- raw二値は既に確率なのでdetach・cloneだけ。raw多クラスはtorch.softmax(dim=1)。確率化はモデルごとの昇順で行う。
- model確率は0～1、多クラスabs(row sum−1)≤1e-6。クラス判定のスコアは有限性と形・配置・精度だけを検査する。
- 重みはbuiltin int/float（bool除外）、finite・0～1・math.fsumで総和1との差≤1e-12。ID昇順にcopyする。正規化本体は通常sumによる総和で各値を割る。混合の検証で受理値を変えない。
- 混合はPython sum(start=0)で昇順にTensor×Python floatを加算する。stack/reduction・行列積・softmax・clipに置き換えない。
- 二値スコア>0.5の比較、多クラスargmax(dim=1,keepdim=True)後float()。返却は[N,1]CPU float32。
- ラベルは[N]/[N,1]、CPU int32/int64/float32、整数0～K−1。二値abs(prob.view(−1)−label.view(−1).float()).mean().item()、多クラス1−gatherした正解確率のmean().item()をfloatへ変換する。
- tensor処理のpublic関数はno_gradで計算し、入力へinplace操作しない。返却tensorは入力storageと結合しない。呼出元の勾配有効状態は復元される。
- 上流接続: raw controller snapshot→normalize→mix/classify→同prob/labelからloss→controller.update(normalized snapshot)。重み・ラベルの状態管理をこのmoduleへ移さない。

## Error Handling

TypeError/ValueErrorで項目と理由を示す。全入力を検査してから計算し、不正時にも入力・外部状態を変更しない。未対応dtype/deviceはcastせず拒否する。torch資源枯渇を握りつぶさない。

## Testing Strategy

- 旧staticへbinary/multi、N=1/複数、ID順・負ID、ゼロ重み、閾値隣接・同率、正規化差を直接照合し、Tensorはtorch.equal、float損失は完全一致。
- 逐次加算の丸めを含むモデル確率→混合→判定が自己拒否しない回帰を置く。判定で再正規化・clipしたら失敗する。
- dtype/device/layout/shape/finite/ID/weight/labelの不正入力、入力のコピー・grad/RNG不変を確認。
- 新上流controllerと旧SwitchingExpertRouterを20以上の標本で接続し、予測・損失・全状態を各標本後に完全一致で確認する。
- 新src全AST検査、全refactoring、schema・旧golden・最終goldenを含む全tests、独立プロセスpublic API smoke。部品以外の完成を主張しない。
