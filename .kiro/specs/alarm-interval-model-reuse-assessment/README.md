# 警報区間での保有モデル再利用評価

正本はrequirements.md（要求r2）、design.md（境界r2）、naming.md（名前r2）、spec.json（承認）。brief.mdは目的/開発ゲート、research.mdはコード根拠、review.mdは独立レビュー/採否。3段階とも実Luna承認済み。tasks.mdとintegration-validation.mdはまだ作成していない。

区間切出しとsession開始の間に必要な未移植部品。既存損失評価/警報区間履歴基準を使い、初期値選択用の評価済み候補と再利用候補を返す。区間の取得、帰属変更、標本吸収、候補生成/開始、通知、旧実装修正は含めない。

2026-10-08ユーザー指示に従い設計/命名レビューの切れ目で停止。source/test/新tasksは未着手。再開時はspec.json/review.mdと上記正本を読み、kiro-spec-tasksでtask案→独立graph sanity→tasks承認→kiro-implへ進む。命名表にないfixture/変数/別名が必要なら実装前に追加命名レビューを行う。リファクタリングの選択規則は旧の保有順を維持し、IMPROVE-001（旧ALGO-001）の性能仮説を混ぜない。
