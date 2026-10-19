# Design Document

## Overview / Boundary Commitments

`sweep/seed_argument_parsing.py`のargparse Actionで入口の表記だけを整数listへ正規化する。core/configuration・sweep plan・実行・保存には新形式を持ち込まない。研究CLIはこの汎用の引数解釈だけをimportし、研究ロジックをsrcへ逆依存させない。

## Components / Interface

公開`SeedListArgumentAction`はargparse.Actionを継承し、`nargs='+'`の値を処理する。単一tokenは従来どおりintへ変換（既存の`+1`等も維持）し、その値が0..2**32-1であることを共通Actionで検査する。範囲はASCII数字の`START-END`だけで、両端も同じ共通Actionで0..2**32-1、昇順であることを検証する。したがって4 CLIとも、単独値・範囲両端のuint32検証を個別CLIの後段へ委ねない。既定値は既存の有効な整数listだけなのでそのまま使う。終端込みで展開する前に個数を算出し、累計100000個以内を確認する。重複は黙って除去せず拒否し、順序は変えない。

解釈・検証は非公開helperで行い、ValueErrorをActionからArgumentErrorへ変換する。namespaceへは検証済みのlist[int]だけを設定する。省略時の既定listは既存値をそのまま使う。複数回の`--seeds`は従来のstoreと同じく最後の指定を採用し、別の暗黙append規則は足さない。

接続先は通常sweep parser、`run_behavior_fix_comparison.py`、`tools/research_prediction_initial_mass/run.py`、`tools/research_component_ablation/analyze.py`の4か所。旧`type=int`を外してActionを指定する。各CLIの追加検証と副作用防止は維持する。既存READMEの引数を読み戻すparserも同じActionを通るため表記差を正規化できる。

## Validation / Traceability

| 要求 | 証拠 |
| --- | --- |
| 1 | argparse単体の列挙/混在/等端/順序/default、既存CLI列挙対範囲のprint-plan一致、同一識別子・再開候補、4接続先のテスト |
| 2 | 不正値・逆順・重複・上限境界/過大範囲を単体検証し、通常/研究CLIの実行前拒否と出力未作成を確認 |

依存境界へ新しいstdlib moduleと既存CLIからのAction importだけを許可する。全task後にLuna highの独立実装レビュー、共用fresh smoke、全pytest（旧11/最終3golden含む）、研究CLI専用checks、Ruff/Pyrightを検証する。NNを新たに掃引する必要はない。整数化後の実行経路は変えない。
