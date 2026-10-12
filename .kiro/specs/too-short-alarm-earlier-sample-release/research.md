# Research & Design Decisions

## Summary
- Feature: too-short-alarm-earlier-sample-release（LEGACY-002の修正）。2026-10-13、ユーザーが採用を決定（LEGACY-016の次）。
- 進め方は、[pending-model-registration-by-own-id](../pending-model-registration-by-own-id/tasks.md)と同じ（旧と同じ扱いを設定で残す、基準版を足す、影響を測る）。そのspecの仕様レビューで出た論点（判定の時点、観測の手段、基準版の照合の2か所、既定のbuilderを通る照合、taskの順）を、最初から設計へ入れた。

## Research Log
### 発生の頻度（2026-10-13、`ab10ad4`、読取りだけの計数。提案構成、client 10、標本2500、集約間隔50）
| 条件 | 短い警報 | 2回目の取込み（次の警報の準備で／容量を超えた確定で） |
|---|---|---|
| sine2 seed 0 | 65 | 1363／308 |
| sine2 seed 1 | 79 | 1758／378 |
| mnist2 seed 0・1 | 1・1 | 4／3、4／2 |
| circle2 seed 0・1 | 1・0 | 0／7、0 |
| sea4 seed 0・1 | 0・0 | 0 |

別のモデルへの2回目の取込みは、観測した範囲で0（短い警報は、現行モデルを変えない。次の警報の準備も、現行モデルへ取り込む）。

### 基準版2のケース（2026-10-13、`200cd73`、runの終わりの学習標本から数えた）
`fedsda-sine2-events`: 学習標本4897、2回目の取込み487、短い警報18。`fedsda-sine2-reference-reuse`: 4900、467、19。ほかの12のFedSDAのケースは、短い警報が0（Git管理外の、その場の計数scriptで数えた。同じ数は、このspecのtest——全体runのtestと、版の差の説明のtest——が、数え直して確かめる）。

### コードの確認
- 保留の位置のowner（`PendingTrainingAssignmentBuffer`）は、設定（`TrainingDataAssignmentSettings`）を持ち、容量をpropertyで返す。完了処理は、ownerを受け取る。方針を、ownerのpropertyで読めば、完了処理の引数を増やさずに済む。
- `AlarmResponseCompletion.drained_pending_sample_indices`を読むのは、完了処理のmoduleの中だけ（`grep`）。
- `TrainingDataAssignmentSettings`は、FedSDAだけが使う（FedDrift・FedAvgのsourceに、参照は無い）。`configuration/run_settings_validation.py`は、容量の項目を読む（項目を足しても、読む側は変わらない）。

## Design Decisions
- 前区間だけを外し、変化区間を残す（ユーザーが採用した案。変化区間は、まだ、どのモデルにも取り込まれていないので、残して、次の判定へ回す）。全部を現行へ確定して消す案は、変化の始まりの標本を、古いモデルへ取り込むことになるので、採らない。
- 検出器の扱い（短い警報のあと、証拠を破棄して再開する）は、変えない（IMPROVE-006。この修正で、短い警報の繰返しが、同じ標本の取込みを増やさなくなるので、その後で、頻度を見て判断する）。
- 方針は、保留の設定に置く（保留の扱いの方針で、設定型が既にある）。

## Risks & Mitigations
- 既定の扱いの正解が実旧に無い → 状態の直接のtest、2つの扱いの結果の比較、全体runの不変条件（2回目の取込み0）、独立レビュー。
- 基準版3で、既存のケースが、必須の経路を通らなくなる → 要求3.5と、seedの探し方を、設計に決めてある。
- 旧との照合が、気づかずに既定の扱いで実行される → 設定の項目に既定値を置かない。照合の入口のassertを、この方針にも広げる。

## References
- [LEGACY-002の記録](../../../docs/research/implementation-findings/legacy-002-retained-assigned-samples-after-short-alarm.md)、[IMPROVE-006](../../../docs/research/improvement-candidates/improve-006-keep-detection-evidence-after-too-short-alarm.md)
- 利用skill: kiro-spec-init/requirements/design/tasks、kiro-impl（--review off）、kiro-validate-impl。共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
