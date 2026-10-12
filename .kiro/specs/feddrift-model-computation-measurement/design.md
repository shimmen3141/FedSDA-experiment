# Design Document

## Overview

共通measure_model_computationをFedDrift全体runへ接続する。新入口はFedSDAの計測入口と同様に参加者設定を受け、毎回新しい実factoryを作る。通常runを複製せず同じ内部実行を使用する。

## Boundary Commitments

計測区間、成功時の境界snapshot、不変な結果と時刻別計数を所有する。学習/検出/サーバ数値処理・共通hookの変更は所有しない。サーバ算術と正規化は別責務として後続へ残す。

## Architecture

`feddrift_measured_run_execution` → `feddrift_stream_run_execution`の非公開共通実行 → 実factory/client/server。共通のmodel_computation_measurementへ依存する。公開の通常入口の引数/戻り値/検査は変更しない。内部実行だけが非公開の任意checkpoint callbackを受ける。callbackは文字列の境界名を受け、計測側がその時点のmeter snapshotを控える。callbackは本番の計測closureだけが渡し、公開APIや方式差替えの窓口にしない。

## Components and Interfaces

```python
@dataclass(frozen=True, kw_only=True)
class FeddriftTimestepModelComputationCounts:
    timestep_index: int
    sample_processing_model_computation_counts: ModelComputationCounts
    synchronization_model_computation_counts: ModelComputationCounts

@dataclass(frozen=True, kw_only=True)
class FeddriftMeasuredRun:
    run_result: StreamProtocolRunResult
    participants: FeddriftRunParticipants
    model_computation_counts: ModelComputationCounts
    preparation_model_computation_counts: ModelComputationCounts
    timestep_model_computation_counts: tuple[FeddriftTimestepModelComputationCounts, ...]
    finalization_model_computation_counts: ModelComputationCounts

def execute_feddrift_stream_protocol_run_with_computation_measurement(
    *, execution_settings: FeddriftStreamProtocolExecutionSettings,
    run_participant_settings: FeddriftRunParticipantSettings,
) -> FeddriftMeasuredRun: ...
```

記録は既存FedSDA計測結果と同様の不変dataclass。設定のexact型とfield再検査は実factoryと既存run入口の契約へ委譲する。記録の手動構築に追加の新契約を置かない。

### 更新順と計測境界

1. 実factoryを構築し、measure_model_computation区間へ入る。設定不正の既存失敗契約を維持する。
2. 既存のconfiguration_validation→乱数準備→isolated CPU torch→initial_preparation（生成器と実factory準備）の成功後に`preparation`を控える。準備の検査/モデル構築/事前学習はこの別区分。
3. 参加者検査、概念とstream生成、時刻ごとの全client標本block処理の順は維持。全client block成功後、通常同期直前に`sample_processing`を控える。
4. `synchronize_timestep`はprepare→R回（全client train→集約/配布）をそのまま実行。成功eventを記録した後に`synchronization`を控える。Rは時刻列の件数を増やさない。
5. 全体終端の成功event後に`finalization`を控える。未処理tailは生成されたまま処理しない。
6. hook区間を抜け、factoryの実参加者を返す。隣接snapshotを共通subtract_model_computation_countsで差分化。準備以後の全合計＝全時刻のsample_processing+synchronization+finalization。ゼロ時刻も準備→終端の差分。

callbackは乱数/モデル操作を行わず、成功境界でのsnapshot読取りだけ。例外を捕捉・包み直さず、共通contextmanagerのfinallyでhookを外す。失敗runの途中計数を成功結果として返さない。通常入口はcallback無しで同じ内部関数を使う。

### 共通単位の適用

FedDriftの独立構造はModelOwnedHiddenLayersなので共有部0、概念固有部に隠れ層＋出力LinearのMACが入る。推論/学習はgrad状態で共通判別し、逆伝播は共通の層形状見積り。クロス評価はclient計算。パラメータの加重平均はhook対象外でありサーバ/各clientの算術計数を0と見せかけるsummaryを作らない。

## Requirements Traceability

| 要求 | 実装と検証 |
|---|---|
| 1.1–1.4 | 新計測入口/結果、共通hook、準備分離、独立MLP算式 |
| 2.1–2.4 | 私有checkpointと時刻記録、全field合計、R/ゼロ同期/tail |
| 3.1 | 通常/実旧の全状態・局所乱数・基本指標照合 |
| 3.2 | 各stage失敗注入、hook集合復帰、正常run再実行 |
| 3.3–3.4 | synthetic/MNIST・SGD/Adam、fresh/exact境界、全回帰 |

## Testing Strategy

実旧oracleは既存run_legacyを使用し、実旧全体入口を先にbootstrapする。REDは実装入口stubから得てcollection失敗をREDに数えない。小モデル1層あたりn×in×out、逆伝播は重み勾配と入力勾配の実有無、update数はR×検出chunk×updates_per_sample×学習可能な保有モデル数から独立確認する。多モデル時は実旧の全状態照合と各計数の保存/再現/合計確認を組み合わせ、旧用途別counterとの単純な同名比較は行わない。共用fresh scriptのFedDrift全体flowを計測入口へ接続する。依存境界は実importのみexact登録。全pytest/golden・Ruff/format・Pyright・names/progress・source identity後に完了/pushする。
