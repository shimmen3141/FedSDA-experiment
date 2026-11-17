# Design Document

## Overview / Boundary Commitments
v5をv6へ派生し、外部依存を加えずinline SVG/CSS/JSを維持する。旧ファイルと実験コード、共有steeringは編集しない。Python公開APIなし。フェーズ5のepochとコマ列は混合v5と一致させる。

## 表示（要求1）
stageの常設border-leftを0にし、paddingで文字位置を保持。現在枠は内側3pxを四辺均一に描く。旧CSSを増やすだけでなく、不要な凡例/拡大用ルールも除く。
headerはh1=FedSDAと右端の補足ボタン。凡例DOMを削除。viewerの末尾20pxを図とtimelineの間の空白行へ移し、timelineを6行目に配置する。狭幅も同じ構造とし全図のみを表示する。
補足はFIFO外標本の模式表示/吸収後解放、モデルID対と固定写し、w/vとT共用、将来標本/ラウンド境界が別時点という誤解防止に絞る。native dialogでshowModal/closeを用い、外側clickはdialog自身がtargetかつ座標がパネル外の時のみcloseする。Escape/明示閉じる/フォーカス復帰も検証。開く時は再生を止める。

## 予測・送り（要求2）
予測select/onchange/overridesと単独出力線を除去。buildのP列はP01〜P07固定。dataset.modeは検査用にmixtureとする。mode条件による説明差し替えを除く。通常学習等のlocal-focusとR10のsync-focusは別目的なので維持する。
move(+1)は末尾でindex=0、全体index=-1からも0へ。prevはindex<=0でdisabled、nextは常時enabled。再生intervalの末尾停止は変更しない。

## 見出し（要求3）
4見出しにsection-headingとdata-heading-actorsを付ける。renderで全体または対応actorのいずれかがactiveなら見出しactiveとする。濃度は1/.28。常時境界として残すクライアント/サーバ名は対象外。
対応: 「同じモデルIDの2状態：左が配布時、右が更新用」=backbone/models/param-sync/param-local/probabilities、結合=prob-sync/prob-local/model-weights/combine-sync/combine-local/mixture-weights、ストア=stores、候補=candidate/references/loss-log/decision。

## 検証
既存のCDP起動基盤をv6専用へ複製し、旧版には書き込まない。18組合せの期待コマ列は混合v5相当の独立列と比較。3画面は1280×720/390×844/1440×900。全体fit/確率文字/通信/FIFO/入力濃淡を維持し、新UIの表示とfocus/実クリック/キー/末尾/自動停止を検査する。代表画像は全体・各機構・モーダル。独立CLIによる仕様レビュー通過後に実装し、全task後に別sessionで実装レビュー。

各画面サイズで全18組合せ（警報3×応答2×検証結果3）を列挙する。各組合せの全コマで期待列一致・固定bbox・濃淡・FIFO/通信/コピー・確率文字・現在枠を確認する。見出しは対応actor集合との一致を全コマで確認する。各組合せの末尾と先頭の送りも確認し、モーダル外クリック/内側維持/Escape/フォーカスと自動停止は各画面で一度確認する。代表PNGは全体/P01/P04/N04/C01/補足を画面別に保存する。
