# 新CPU接続と依存境界の証拠

2026-10-08、Task 4。承認済みのruntimeと隣接公開ownerを使う。

## 依存境界

- 実sourceの23 import symbol、AST guardの許可集合、命名表のexact一覧は一致した。実ASTから集合を抽出して照合した。
- 両resolverのexact対象へ登録済み。235注入条件でdirect/alias/relativeを許可し、wholemodule/private/star/child/上位の別層依存を拒否する。
- 対象64＋全AST2455＝**2519 passed、20.67s、exit0**。

## 新CPU process

元checkoutの`venv/refactoring-tests/alarm-buffer-response-fresh-cpu.py`を別processで実行した。旧実装/test moduleをimportせず、実分類器とoptimizer、標本・統計・計数・評価・帰属・FIFOの新ownerだけを生成する。

2/4class×正規ID2/一時ID−3×不足/現行維持/activeの**12条件成功、exit0**。active sessionは公開区間解決で作り、実応答へ接続した。

- 全条件でFIFO状態を維持し、不足だけ消費判断False、その他True。
- 不足で前区間3件、十分とactiveで全5件の学習標本/統計更新、現行ID維持。
- activeは同じsession参照、区間件数・最小件数・評価owner・Python Random・候補設定・元位置をobjectにしても成功（未使用）。
- torchとグローバルPython乱数は不消費。明示Randomは正規IDでの前区間評価保存だけ消費する。

初回はPYTHONPATH未指定で新packageのimport前に失敗した。成功として扱わず、次の明示設定で再実行して上記を確認した。

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
../../venv/Scripts/python.exe ../../venv/refactoring-tests/alarm-buffer-response-fresh-cpu.py
```

外部証拠: `alarm-buffer-response-fresh-cpu.log`、`alarm-buffer-response-task4-tests.log`、`alarm-buffer-response-import-audit.json`（元checkoutのvenv/refactoring-tests）。全NN分岐と勾配/履歴/NumPyの検証は本specの対象testを参照する。新全体runの数値回帰はまだ主張しない。独立レビューはreview.md/spec.json。
