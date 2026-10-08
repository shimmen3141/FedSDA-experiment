# 具体的な改善候補

コードを調べて見つけた、範囲の限定された簡略化・効率化・アルゴリズム調整案を1候補1ファイルで記録する。旧実装・新実装のどちらも対象。各ファイルが詳細と状態の正本で、本書は入口と記録規約。

## 記録先の選び方

| 記録先 | 対象 |
| --- | --- |
| improvement-candidates/ | 複雑な処理の簡略化、計算量/通信量/メモリ削減、限定的な選択規則・設定の調整、その他具体的な改善案 |
| [implementation-findings/](../implementation-findings/README.md) | 不具合の疑い、契約・期待挙動との不整合、再現条件と影響の調査 |
| [研究バックログ](../research-backlog.md) | 広い研究方向、代替設計、アイデア段階の研究・実験候補 |

区別が難しい場合は、期待する契約との不整合が疑われるならimplementation-findings、改善余地という仮説なら本フォルダへ置く。未確認と明記し、同じ事象の詳細を二重に書かず相互リンクする。分類を変えた場合も元IDと移動先を記録する。既存のLEGACY記録は今回一括移動しない。

## 一覧

| ID | 件名 | 状態 | 記録 |
| --- | --- | --- | --- |
| IMPROVE-001（旧ALGO-001） | 警報区間の同率再利用候補で現行モデルを優先 | 未検証・未採用 | [詳細](improve-001-current-model-priority-on-reuse-ties.md) |
| IMPROVE-002 | 参照モデルの固定で不要なランダム初期化を省く | 未検証・未採用 | [詳細](improve-002-skip-random-initialization-of-fixed-reference-models.md) |
| IMPROVE-003 | 固定した参照モデルの検証損失を到達時にまとめて評価 | 未検証・未採用 | [詳細](improve-003-batch-fixed-reference-loss-evaluation.md) |
| IMPROVE-004 | 履歴基準が使えないモデルの警報区間評価を省く | 未検証・未採用 | [詳細](improve-004-skip-alarm-loss-evaluation-without-reuse-baseline.md) |
| IMPROVE-005 | 前区間の事前検査と吸収で損失評価を共用 | 未検証・未採用 | [詳細](improve-005-reuse-prevalidated-alarm-interval-losses.md) |
| IMPROVE-006 | 変化区間が不足した警報の後、検出の証拠を破棄しない | 未検証・未採用 | [詳細](improve-006-keep-detection-evidence-after-too-short-alarm.md) |
| IMPROVE-007 | 最終予測に不要な比較予測診断を選択実行する | 未検証・未採用 | [詳細](improve-007-optional-counterfactual-prediction-diagnostics.md) |

## 記録・更新の規約

IDはIMPROVE-連番、ファイル名は`improve-<番号>-<対象と案>.md`。次はIMPROVE-008。候補の発見時点で記録し、実験済みである必要はない。

各ファイルへ以下を記載する。未確認・未定の項目はそのまま明記する。

- 日付、対象コード/構成、基準commit、発見したspec。
- 種別（簡略化・計算効率・通信効率・メモリ・アルゴリズム/設定調整等）と状態。
- コードで確認した事実・具体的な箇所。効果の仮説とは分ける。
- 変更案、期待する利点、悪化やトレードオフの懸念。
- 挙動を維持する整理か、数値/選択/RNG/診断が変わる可能性のある変更か。
- 必要な比較・指標・採否基準、関連する不具合/研究案/spec。
- 検証結果、採用/保留/見送りの理由、変更commit。結果が否定的でも削除しない。

状態は「未検証・未採用」「検証中」「保留」「採用」「見送り」。候補の記録は変更・golden更新・長時間実験の承認ではない。旧挙動を維持する移植へ混ぜず、採用時は変更の影響範囲に応じた別spec/検証で扱う。
