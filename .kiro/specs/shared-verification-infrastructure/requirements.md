# 検証基盤の整理 — 要求 revision1

リファクタリングの主担当が、specごとに検証用のscriptやtestを書き足さずに済み、かつ検証が弱くならないよう、共用の検証を3つ整える。

## 境界

testとtest用scriptだけを追加・変更する。`src/`、旧実装、goldenは変更しない。依存testの既存の登録（moduleごとの許可集合）と既存の注入契約testは移さず、消さない（2026-10-09ユーザー決定。[IMPROVE-009](../../../docs/research/improvement-candidates/improve-009-unify-dependency-boundary-tests.md)）。新しい接続の移植、標本1件の処理、新全体runは除外する。

## 1. 新実装だけの流れ

- 1.1 The Fresh Process Smoke shall 旧実装とtest moduleをimportしない別のprocessで、接続済みの流れ（警報1回ぶんの処理の5種類の結果、候補検証中の警報、候補検証の進行と確定の4種類の結果、確定の後の次の警報）を、1つの保持・適応記録・診断証拠・損失監視・保留位置のownerで実行し、各段の後のownerの状態が互いに合うことを確かめる。
- 1.2 When 全pytestを実行したとき, the Fresh Process Smoke shall 上の流れを、pytestのprocessが読み込んだmoduleを引き継がない別processで実行し、失敗・旧実装やtest moduleの読込み・必要な結果の未観測のいずれかがあれば失敗する。
- 1.3 The Fresh Process Smoke shall Git管理下に置かれ、worktreeルートから単独でも実行できる。

## 2. 依存の許可集合

- 2.1 The Dependency Boundary Test shall moduleごとに登録した許可集合のすべての名前が、そのmoduleの実際のsourceでimportされていることを確かめる（`__future__`を除く）。使っていない名前が許可に残っていれば失敗する。
- 2.2 The Dependency Boundary Test shall 許可集合の読取りが空振りした場合（登録の書き方が変わって読めなくなった場合）に失敗する。
- 2.3 The change shall 2.1で見つかった、使っていない許可を外す。

## 3. testの穴（NEW-002）

- 3.1 The Held Candidate Validation Progress Test shall 保持への反映が、応答と保持のownerの派生型を拒否することを確かめる。
- 3.2 The Held Candidate Validation Progress Test shall 終端回収が、適応記録を追加してから保持を解除することを確かめる。

## 4. 維持

- 4.1 The change shall `src/`・固定旧実装・goldenを変更せず、既存の全回帰を維持する。
