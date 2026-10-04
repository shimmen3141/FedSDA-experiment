# 設計: 候補パラメータ初期化
## Overview
初期化元の方式から完成した独立parameter snapshotを作る。手法固有の最小損失選択と平均は、この一出力契約の内部にまとめる。
上位は旧get_params相当を新dictへ明示変換する。旧形式読込や互換aliasはproductionに置かない。

## Boundary Commitments
### This Spec Owns
三方式の固定条件宣言、明示入力全検査、初期化元選択、snapshotのdetachコピーと等重み平均。
### Out of Boundary
モデル生成/forward/学習/optimizer/reset/共有backboneへの適用/統計seed/採否/FIFO/ID/通信/新全体run。
### Allowed Dependencies
計算moduleはstdlib、exact torchとtorch.Tensor、同機能candidate_parameter_initialization_settings moduleと公開型だけ。
設定moduleはstdlib、exact core.settings_field_validation moduleとvalidate_settings_field_valuesだけ。
別settings、torch子module/private、旧、configuration/runtime/データ/監視/候補評価/統計は不可。
torch数値操作を必要とするsnapshot作成に依存を限定する。既存候補評価と同様に、手法固有判断をlearning汎用層へ流さない。
### Revalidation Triggers
選択元/tie/空/平均の演算順/copy/型契約を変えたら旧oracleと評価→初期化接続を再検証。適用や学習はそのspecで別検証。

## Architecture
専用frozen設定と一つの無状態関数。汎用modelstate型、abstract factory、初期化plan、モデルcallbackを追加しない。
snapshotはモデルIDとparameter名を保持するdict。modelとparameter順は呼出側から受け取り維持。
shared構成でもbackbone/adapter/headを分割せず完全stateを扱う。
実体への適用は上位の責務であり、今回はテストの明示loss評価→snapshot選択のみ接続する。

## File Structure Plan
- src/federated_learning_experiments/methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py: 固定選択肢を宣言する。
- src/federated_learning_experiments/methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py: 全検査と完成snapshot作成。
- tests/refactoring/test_candidate_parameter_initialization.py: 旧oracle、拒否、独立性、環境、候補評価結果からの接続。
- tests/refactoring/test_single_run_dependency_boundaries.py: 上記2moduleのexact依存許可/禁止注入。
- 対象spec/roadmap: 正本と実測承認。旧発見は実証がある場合だけ既存台帳へ。

## Components and Interfaces
```python
@dataclass(frozen=True, kw_only=True)
class CandidateParameterInitializationSettings:
    candidate_parameter_initialization_source: str  # 必須、metadataで3choice宣言

select_candidate_initial_parameter_snapshot(
    *, settings: CandidateParameterInitializationSettings,
    available_parameter_snapshots_by_model_id: dict[int, dict[str, torch.Tensor]],
    current_training_model_id: int,
    evaluated_mean_losses_by_model_id: tuple[tuple[int, float], ...],
) -> dict[str, torch.Tensor] | None
```
正式choiceはassigned_training_model / lowest_evaluated_mean_loss_model / equal_mean_of_available_models。旧choiceを受理しない。設定省略defaultは作らず、上位が方式を明示する。
exact設定型とpublicconstructorで再検査コピーする。元__post_init__を呼ばない。
全snapshotと評価を選択前に検査する。外側/内側exact dict、IDは負値も可能なexact int、parameter名exact非空str、snapshotは非空。
Tensorはexact torch.Tensor、CPU、dense strided、dtypeはfloat16/bfloat16/float32/float64/complex64/complex128/uint8/uint16/uint32/uint64/int8/int16/int32/int64/bool。
浮動/複素値はfinite。scalar/空dimension/noncontiguousは通常denseとして受理。gradient付き入力はdetachして扱う。
全モデルで同key集合/同keyのshape・dtype・device一致。inner key挿入順が違っても先頭model key順で平均する。
評価はexact tupleのexact pair、ID重複なし、各IDはavailable内、lossはbuiltin int/float（bool不可）有限[0,1]。
current IDは常にexact int、存在は実際にcurrentを選ぶ場合だけ必要。averageのavailable空と評価空ならNone。
assignedはcurrent、lowestは評価のmin key=lossのみ（IDでtie-breakしない）、空評価ならcurrent。
等重み平均はmodel順/先頭key順でtorch.stack(...).mean(dim=0)、非浮動・非複素値は先頭clone。平均の結果もfiniteを確認し、overflowによる非有限値をclipしない。
単一選択のkey順はそのmodel自身の順。戻り値はplain dict、各value.detach().clone()の独立storage、requires_grad False。
snapshot metadata/load_state_dictの責務は今回持たない。parameter名は意味を解釈しない。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1 | assigned選択/非最小current旧oracle |
| 1.2 | 最小loss/評価外除外/同率先着/空fallback/負IDoracle |
| 1.3 | 全保有順stackmean/整数・bool先頭/empty None/複素oracle |
| 1.4 | 全key/value/shape/dtype/device/順序の旧照合 |
| 2.1 | exact型/forged設定/後段invalid/欠落/重複/不整合/finite拒否 |
| 2.2 | 双方向変更/別呼出/storage/gradient detach |
| 2.3 | Python/NumPy/torch RNG、grad、default dtype/meta device保持 |
| 3.1 | 実旧unbound helperと評価mean→明示入力のtest-only接続 |
| 3.2 | exactAST/CPU fresh smoke/全golden/正本・台帳 |

## Error Handling
設定は既存RunSettingsValidationError(ValueError)を使い、型違反はTypeError、値/欠落/構造/結果非有限はValueErrorで項目名を示す。
全検査・全出力構築をローカルで完了する。途中例外で入力や外部モデルを変えない。空averageのNoneは正常。

## Testing Strategy
Task1 TDDで三方式・同率/空/評価外・完全shared風keys/dtypesをunbound旧oracleへ照合。モデルconstructor無しget_params stubだけ。
Task2拒否/後段異常/コピー/gradient/環境、既存警報後候補評価出力のmeanを明示渡して結果を検証。
Task3 AST禁止注入のRED→exact2module境界、fresh CPU smoke、全tests、配置/9条件/旧golden無変更、Luna最終GO。
