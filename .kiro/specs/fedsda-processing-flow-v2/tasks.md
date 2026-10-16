# Tasks

- [x] 1. v2の図とモデル対を実装する
  - [x] 1.1 元HTMLの保全を確認し、短線と2状態モデルを別HTMLへ反映して描画検証する。
  - _Requirements: 1, 2_
- [x] 2. 共通コマ列と分岐操作を実装する
  - [x] 2.1 順序契約の検査を先に用意し、段階列・無効枝・位置保持・別時点の区切りを実装する。
  - _Depends: 1.1_
  - _Requirements: 3_
- [x] 3. 最終描画と独立レビューを確認する
  - [x] 3.1 全組合せ/操作/画面サイズと代表画像を確認し、独立CLI実装レビューを受け、有用な指摘を反映する。commit/pushしない。
  - _Depends: 2.1_
  - _Requirements: 4_

## Implementation Notes
- 既存完成版と元checkoutは保持。新specも未commit。
- 仕様レビューPASS。Minorのepoch切替位置をsample/future/roundと明記して反映。再レビュー不要。
- v2を専用worktreeへ作成。既存HTML・元checkoutはSHA256一致のまま保持。HTMLの外部参照なし、100 IDに重複なし。stage/commit/pushなし。
- 主担当が専用プロファイルのheadless Chromeで108組合せ×2256コマを1280×720、1440×900（通常motion）、390×844で確認、issuesなし。短線21本は小さい矢尻＋実線。通常の長線はtravel、低モーションでは停止。代表画像も確認。
- 最初の起動でrender対象selectorがdiagram/timelineのdata-stageまで拾う不具合を検出し、.stage[data-stage]へ限定して解消。通常motion検査は短線だけのP02から長線のあるP03へ修正。無効段階の追加行はタイトルへ移し高さを保持。全組合せを再検証済み。
- 独立Lunaの実装レビューPASS。日本語出力制約があり、Haikuを試したが週上限429。ASCII JSONと最終画像3枚を読むLunaの補完レビューもPASS。独立実行はJS構文検査、ブラウザは主担当実測を利用。Python数値試験は未実施・対象外。
