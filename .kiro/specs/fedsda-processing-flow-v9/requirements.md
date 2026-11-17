# Requirements Document

## Introduction
v8を維持してv9を新規作成し、クロス評価時に左右のモデル群を同じ単位で把握できるようにする。候補検証の「固定参照」は「既存モデル」という表示に統一する。

## Boundary Context
v9とこのspecのみ。旧版/既存spec/元checkout/実験コードを保持しstage/commit/pushしない。数値/コマ順は変更しない。

## Requirements
### Requirement 1: クロス評価時だけの点線枠
1. When 実際に評価するR06のコマを表示する, the diagram shall 更新用の共有バックボーン/モデル群にも点線の枠を表示する。評価用ストアへ向かう既存の矢印の始点は、この枠の右辺中央に一致させる。
2. While R06以外のコマまたは全体図を表示する, the diagram shall この追加枠を表示しない。受信するR05にも追加しない。
### Requirement 2: 候補検証の表示名
1. The diagram shall 候補検証の「固定参照」を「既存モデル」に改め、関連コマの説明/補足も自然な文へ合わせる。機構内部のreferences IDは維持する。
2. The diagram shall ここでの既存モデルが警報時点の比較用の写しであり、検証中は追加学習しない意味と「警報時点の写し」の副見出しを保持する。
### Requirement 3: 保持と検証
1. The verification shall 3画面の全18経路472コマで追加枠がR06だけに出ること、元の矢印と枠の接続、改名/文字の収まり、既存操作・濃淡・フラッシュ抑制が維持されることを確認する。HTMLのみのため数値試験は対象外とする。
