# AdaHedge診断の証拠状態 — 設計 revision1

## Overview

単一の`AdaHedgeDiagnosticEvidence`をevaluationに配置する。診断用の累積損失とmixability gapを所有し、重み取得・損失更新・概念再始動だけを公開する。Fixed-Shareの予測重み管理へ依存しない。

## Boundary Commitments

- **This Spec Owns:** 単一証拠owner、入力契約、演算順、pool reset/concept restart計数と独立copyによる観測。
- **Out of Boundary:** 通知の発火・複数ownerの保持・真concept選択・集約後再較正・leader/effective countの集計・保存・検出episode・client進行・設定追加・新全体run。
- **Allowed Dependencies:** `math`のmodule importと`collections.abc`の`Iterable`/`Mapping`だけ。`__init__.py`にexportを増やさない。旧実装、runtime、tensor、グローバルconfigへの依存は禁止。
- **Revalidation Triggers:** ID/数値の受理集合、copy公開、計数規則、再始動範囲、演算順を変えたら後続通知・診断集計・集約再較正の接続を再検証する。

## Architecture / File Structure Plan

| ファイル | 責務 |
| --- | --- |
| `src/federated_learning_experiments/evaluation/adahedge_diagnostic_evidence.py`（新規） | 単一診断証拠の数値計算と所有 |
| `tests/refactoring/test_adahedge_diagnostic_evidence.py`（新規） | 実旧oracle、拒否前不変、copy、数値分岐の対照 |
| `tests/refactoring/test_single_run_dependency_boundaries.py`（変更） | 注入契約testと両AST resolverへexact依存を登録 |

既存のFixed-Shareと同じくstdlibによる計算を独立移植する。式を別ライブラリへ置き換えると演算順の一致を保証できないため追加依存は採用しない。1 ownerなので汎用Protocolやイベントバスは作らない。

## Components and Interfaces

引数はkeyword-only。constructorは引数なし。

| API | 入出力・更新 |
| --- | --- |
| `get_diagnostic_weights_before_loss_observation` | `model_ids: Iterable[int]` → `dict[int,float]`。集合が変われば同期する |
| `update_evidence_after_loss_observation` | `observed_losses_by_model_id`、`diagnostic_weights_by_model_id`: `Mapping[int,float]` → None |
| `restart_evidence_after_concept_operation` | 入力なし。累積損失を空、gapを0、concept計数+1 |
| `cumulative_losses_by_model_id` | dict copy |
| `mixability_gap`、`model_pool_reset_count`、`concept_operation_restart_count` | 読取りproperty |

### 入力契約

IDはbuiltin int（bool除外）、負可。Iterableを先にtuple化して非空・重複なしを検査し、昇順へ並べる。更新入力はMappingであることを確認し、両mapを独立dictへcopyしてから全ID・全値を検査する。数値はbuiltin int/float、bool除外、有限floatへ変換。型不正はTypeError、空/重複/非有限/float変換overflow/不正重み/集合不一致はValueError。重みは0～1、`math.fsum`総和の1との差1e-12以内。受理した値は再正規化しない。

検査は集合同期を含む最初の更新より前に置く。入力Iterable/Mappingの読取り中の例外もownerを変更しない。並行更新、破壊された内部私有field、資源不足の巻戻しは保証外。

### 演算と更新順

1. ID集合が現累積損失のkey集合と違えば、昇順IDに0.0を置いた新dictとgap=0.0を計算候補にする。旧dictが非空だった場合のみpool計数+1を候補にする。同集合なら現在値のcopyを使う。
2. 学習率はモデル数1以下またはgap<=0なら`math.inf`、それ以外は`math.log(モデル数)/gap`。
3. 重み取得: 単一は1。無限学習率は最小累積損失の同率IDへ1/件数。有限は`math.exp(-eta*(loss-minimum))`、通常`sum`、その総和で正規化。全計算成功後に同期候補を所有状態へ反映して独立dictを返す。
4. 観測後更新: 昇順IDの損失を`min(1.0,max(0.0,float(value)))`。通常`sum`で期待損失。無限学習率では重み>0のモデルの最小損失がmix loss。有限では重み>0の`log(weight)-eta*loss`を作り、最大値を引いたlog-sum-expでmix lossを計算。gapに`max(0.0,expected_loss-mix_loss)`を足してから、昇順IDで累積損失へ有界損失を足す。
5. 全演算は局所候補だけで行い、成功後にloss dict、gap、pool計数を反映する。再始動はconcept計数のみ増やし、空状態の再始動でも数える。再始動直後の重み取得でpool計数を増やさない。

旧updateは集合同期の後に重み集合を検査するが、新実装では拒否による部分更新を避けるため先に全入力を検査する。成功時の昇順・float化・sum順・gap加算→損失加算は維持する。通常経路の同値性は実旧対照で確かめる。

## Requirements Traceability / Testing Strategy

| 要求 | 検証 |
| --- | --- |
| 1.1 | 独立owner/初期property/再始動後の状態 |
| 1.2, 1.3 | 実旧の取得列、未更新の集合変更、順序違い、単一・同率・有限/無限eta |
| 2.1 | 実旧の複数更新、負ID、損失制限、集合変更を伴う直接update |
| 2.2 | 実旧の空/非空/繰返しrestart、取得後のpool計数 |
| 2.3, 3.3 | 呼出元mapと取得済みcopyの独立、Python/NumPy/torch乱数の状態全体不変 |
| 3.1, 3.2 | populated ownerへの拒否、全4状態の不変、壊れた後続map値と集合変更の組合せ |
| 4.1 | 実旧`AdaHedgeRouter`を直接呼び、floatを許容誤差なしで対照 |
| 4.2 | exact依存guard、旧非import fresh process、固定旧差分、Windows全回帰 |

変異は型/値/総和/集合検査の削除、検査を集合同期後へ移す、昇順省略、gap式/計数/再始動の破壊、copy省略を含める。collection失敗を検出に数えず元byteへ復元する。部品成功と保存診断全体の同一性は区別する。
