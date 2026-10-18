# Requirements Document

## Project Description
候補検証を単純化したい研究者のため、検証なしOの精度低下が新モデルの予測初期重みによるものかを独立した対照で調べる。Dは両基準を上回らなかった。製品フローを複雑化せず切り離せる研究コードとして試す。

## Requirements
### Requirement 1: 初期質量だけの研究介入
1. The diagnostic shall V（検証あり）、O（検証なし）、R（既存比率保持・新ID一様質量）、P（既存比率保持・新ID小質量）を区別する。
2. When R/Pで新ID追加と正の既存残存質量がある, the diagnostic shall ラベルを使わず重みを設計の式で初期化する。分散resetと集合変更計数は現行どおりとする。
3. While 初回、同一集合、削除のみ、既存IDが残らない、または残存質量が0である, the diagnostic shall 元の同期処理を使う。集約のclear/replay・二者混合のT・通常損失更新を変更しない。
4. If runが成功または失敗する, the diagnostic shall 介入関数を復元する。追加の乱数を消費しない。
### Requirement 2: 分離と記録
1. The diagnostic shall tools/research_prediction_initial_mass内だけに実行・明示専用checksを置く。src・共通pytest/CI/fresh/Ruff設定を変更しない。
2. The diagnostic shall 各runの完全設定、mode、目的、code commit/dirty、source/script hash、環境、実指標、全NPZ、介入前後重みとID/T、RNGを別の診断形式へ保存する。既存形式・結果を変更しない。
3. If 出力が既存または引数が不正である, the diagnostic shall 学習前に拒否する。計画表示は読取り専用、worker1run/process・各1thread・上限14とする。
### Requirement 3: 対照と分析
1. The diagnostic shall V/Oが直接実行と配列/指標/RNGで一致することを確認する。
2. The analysis shall R/P対Oで学習・適応・通信・最終パラメータが不変か確認し、差があれば予測だけの効果とは扱わない。
3. The analysis shall V/O/R/Pの精度・上下り/合計通信・介入数・seed依存を示す。小規模実測を本規模の採否に使わず、自動採否を行わない。
