# Research

## 版7の新しいケースの条件の下調べ（2026-10-20、Windows）

`fedsda-sine2-events`・`fedsda-mnist2-decisions`の版6の設定から、採否の方針だけを検証なしへ替え、候補の共同更新を`update_shared_features=False`へ差し替えて（研究ツールと同じ形。sourceは変えていない）、seedごとに経路を数えた。script: `C:\Users\yshin\AppData\Local\Temp\claude\c--Users-yshin-Research-FedSDA-experiment\799b882d-c886-45dd-a18e-a06f3e20f4b4\scratchpad\v7_case_probe.py`（gitの対象外。結果は同じ場所の`v7_case_probe.json`）。

| ケースの元 | seed | 採用 | 学習帰属の切替え | 再較正の標本 | 統合 | 登録 |
|---|---|---|---|---|---|---|
| sine2（3 client、1500標本、間隔50） | 0 | 6 | 34 | 2233 | 3 | 4 |
| | 1〜5 | 4〜9 | 17〜27 | 2116〜2382 | 2〜11 | 4〜9 |
| mnist2（2 client、100標本） | 0 | 4 | 4 | 106 | 0 | 2 |
| | 1〜3 | 2〜4 | 2〜4 | 41〜74 | 0〜2 | 2 |

- どのseedでも、登録されないままの採用と、同じ標本の2回目の取込みは0。
- 新しい2ケースは、元のケースと同じseed 0にする（sine2は採用・切替え・再較正・統合が全部0でない。mnist2は統合が0なので、統合はsine2だけで求める）。sine2では、切替え（34）が採用（6）より多いので、保有モデルの再利用も通る。
- 実装後の版7の値は、差し替えではなく新しい設定で取り直す（同じ値になるはずだが、goldenは実装したsourceから生成する）。
