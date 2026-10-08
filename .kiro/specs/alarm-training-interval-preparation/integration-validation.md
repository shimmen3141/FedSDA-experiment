# 警報時の学習区間の準備 — 統合検証

## 対象と境界

区間分割と前区間の評価保存→吸収を、新しい3moduleで組み立てた。Task 3の実装/test commitは`8942c41`。Task 5で依存guardの同じ許可条件の重複を除去した。全体検証の対象commit/hashとfeature最終判定は、全回帰後に記録する。

旧`_resolve_drift`の対応区間を実NNで実行するoracleを使用。推定区間長、検出器reset、適応eventの記録はこのspecの範囲外として差し替えた。保存、吸収、区間解決、共同更新、候補学習・観測の実計算は差し替えていない。

## task証拠

| task | 主担当の実測と証拠 |
| --- | --- |
| 1 record | 実装前ImportErrorのRED、依存注入22 failed/28 passed。登録後対象＋AST2067 passed。独立Luna承認 |
| 2 runtime | 実装前ImportErrorのRED、依存注入49 failed/155 passed。対象98＋AST2220＝2318 passed。同じ分類器を事前評価・吸収が参照することを追加検証。独立Luna承認 |
| 3 接続 | 152 passed、主担当13.37s、独立Lunaも152 passed。2/4class×再利用・維持・候補開始、実共同更新と候補観測、最小件数直前・一致 |
| 4 検出力 | [6変異すべて検出、byte復元後152 passed](mutation-evidence.md)。独立Luna承認 |
| 5 依存境界 | 最終対象152＋AST2220＝2372 passed / 10.05s / exit0。独立レビューは別記 |
| 6 fresh CPU | 旧/test importなしの新process、2/4class×正規ID2/負ID−3の4条件成功 / exit0。独立レビューは別記 |

### exact依存集合

入力recordは`dataclasses.dataclass`と`ObservedTrainingSample`の2symbol、結果recordは`dataclasses.dataclass`と`IndexedObservedTrainingSample`の2symbol。runtimeは命名・設計にある15symbolのみ。実sourceの相対importを絶対symbolへ解決し、guardの許可定数集合とのexact一致を確認した。ImportFromの直接symbol resolverとImportのwholemodule拒否resolverの両方に3moduleを登録。注入204条件にdirect/alias/relativeの許可とwholemodule/private/star/child/upwardの拒否を含む。

監査で同じ結果recordの許可分岐が2回あったことを発見し、到達しない重複1回を削除した。許可集合・実装挙動は変わらない。監査scriptと報告は外部`alarm-preparation-audit.py`と`alarm-preparation-dependency-audit.json`。

### fresh新CPU

このPCの元checkoutの`venv/refactoring-tests/alarm-preparation-draft/fresh_cpu_smoke.py`を、worktreeの`src`をPYTHONPATHに指定して実行した。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONPATH=(Resolve-Path src).Path
../../venv/Scripts/python.exe ../../venv/refactoring-tests/alarm-preparation-draft/fresh_cpu_smoke.py
```

実残差アダプタ分類器、Adam状態、5標本（FIFO容量4）、前区間3/変化区間2を生成。前区間のtraining標本数3・損失件数3・概念計数（Noneを除外）・変化開始位置3と借用tuple、FIFO不変を確認。正規IDは評価標本2件を保存し明示Randomを消費、負IDは保存/明示Random消費なし。torch/global Python乱数は不変。`sys.modules`に旧実装/testがないことも確認した。別PCでは、この手順・入力から新公開APIだけで再作成できる。

## 要求との対応

| 要求 | 実行証拠 |
| --- | --- |
| 1.1 | 短いspanで前区間と末尾変化区間の位置・payload・概念IDの順と対応を旧と照合 |
| 1.2 | 保留件数と等しい/長いspanで前区間空・全件変化区間 |
| 1.3 | 空入力で標本/統計/計数/評価保存/乱数不変 |
| 1.4 | frozen/kw_only・tuple、借用Tensor、FIFOと最終位置不変、容量+1 |
| 2.1 | 保存件数・抽出順・容量・Python乱数を実旧と比較、保存→吸収の実呼出し順 |
| 2.2 | 負IDは評価保存と抽出乱数を省略し吸収 |
| 2.3 | 前区間空でno-op |
| 2.4 | training/全体・クラス損失/概念計数の一致、None除外、他モデル/parameter/grad/optimizer/帰属不変 |
| 2.5 | 吸収済みownerを実区間解決へ渡し、解決/後続更新/候補観測まで実旧と比較。最小件数直前・一致でも前区間処理 |
| 3.1 | 欠落/余分/重複/逆順の位置を全状態・乱数不変で拒否 |
| 3.2 | owner・型・値・形状・末尾の不正前区間を、保存を含む全更新前に拒否。同一分類器の事前評価→保存→実吸収 |
| 3.3 | 実旧対照・Python乱数最終state、torch/NumPy不変、変異・依存・fresh CPU・全回帰 |

## 保証範囲と残る制約

- 警報制御全体のactive session経路、最小件数による中止、FIFO消費/保持、検出器reset、適応event・通知・再利用計数、active session保持、新client/全体runへの接続は後続spec。**新全体runのgolden一致は未検証**。
- 入力起因の失敗は評価保存前に拒否するため、前区間を損失評価後、実吸収でも再評価する。追加forwardのコストを含み、計算量診断への接続前に判断する。アルゴリズム改善として勝手に省略しない。
- 明示位置とpayloadの意味上の対応は供給側が保証する。borrowしたTensorの内容は外部から変更できる。並行更新、私的状態の破壊、OOMや検査後の予期しない失敗のrollbackは保証しない。
- CPU float32の既存分類器が対象。変化区間の分類器依存検査は後続へ残す。複数回適用による二重吸収を防ぐ責務も呼出側にある。
- 固定旧goldenの成功と、新部品の実旧対照を別々に記録する。旧の不足データ時のFIFO保持などの修正候補は今回変更していない。
