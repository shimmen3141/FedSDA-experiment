# Development finding: 既定device復元の検証コードがcontextを残す

- 観測日: 2026-10-06
- 観測した作業: classifier-per-sample-bounded-loss-evaluation Task1独立Lunaレビュー
- 改善先: project（検証コードと状態保護のレビュー）
- 関連artifact: ../.kiro/specs/classifier-per-sample-bounded-loss-evaluation/review.md、tests/refactoring/test_classifier_bounded_loss_evaluation.py

## 観測した事実
testで既定deviceをmetaへ変更し、finallyで元のget_default_device()へ戻したところ、戻り値はcpuでもTorchのfunction mode stackにはDeviceContextが残った。対象77testsは成功しており、通常の値比較だけではこの漏れを検出できなかった。Lunaが指摘し、主担当が別processで再現した。

既存固定Torch2.12.1+cpuのfresh processで以下の確認を行った（private stack APIは調査だけに用い、productionへ依存させない）。

```python
from torch.overrides import _get_current_function_mode_stack
before = len(_get_current_function_mode_stack())
original = torch.get_default_device()
torch.set_default_device("meta")
torch.set_default_device(original)
```

stackの件数は0→1、get_default_deviceはcpu。別fresh processのwith torch.device("meta")ではcontext内の空Tensorがmeta、終了後stackは0→0である。

## 影響とworkaround
- 検証コードが後続testのdispatch環境を変更する。値が正しいだけでは状態復元を証明できない。
- scoped device contextへ変更し、dtypeだけをfinallyで復元した。主担当の対象77 passed/3.41秒/exit0、Ruff/format成功。
- 旧研究アルゴリズムの不具合ではなく、新しい検証fixtureの問題。過去研究実験への影響を示す証拠はない。

## 仮説と改善案
- 仮説: setterでdevice値を戻すことと、元のfunction mode stackを復元することが異なる。
- 今後の一時環境変更testはスコープを使い、getterの値だけで副作用なしと判断しない。既存tests/refactoring内のset_default_device検索は今回の修正後に該当なし。

## 改善結果
新testでsetterを除去し、with torch.device("meta")のcontext内でも評価結果がCPU float32となることを既存の入出力契約と一緒に確認した。本体のAPI/数値処理は変更していない。Luna再レビューの証拠は対象specへ記録する。
