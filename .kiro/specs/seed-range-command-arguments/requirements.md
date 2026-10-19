# Requirements Document

## Introduction

実験担当者はseed列挙を短く記述したい。既存の整数列挙を維持し、`--seeds 0-9`と混在指定を各seed対応CLIで使えるようにする。

## Boundary Context

通常掃引、挙動比較、予測初期質量の研究CLI、component ablation解析のseed引数だけを対象とする。他の数値軸の新構文、実験アルゴリズム、run設定/識別子、記録形式は変更しない。

## Requirements

### Requirement 1: 列挙と範囲を同じseed列へ展開する
**Objective:** 研究者として、列挙と範囲を必要に応じて混在させたい。
#### Acceptance Criteria
1. When `--seeds 0-9`を指定するとき、CLIは0から9を終端込みで指定順に展開する。
2. When `--seeds 0-2 5 8-9`を指定するとき、CLIは`0 1 2 5 8 9`と同じ整数列を渡す。
3. When 従来の整数列や既定値を用いるとき、CLIは既存の順序と値を維持する。
4. The CLI shall 範囲・列挙の違いでrun設定・識別子・再開候補の一致を変えない。

### Requirement 2: 誤指定と過大な展開を拒否する
**Objective:** 研究者として、重複や誤った範囲による意図しない実行を避けたい。
#### Acceptance Criteria
1. If seedが非整数・負・2の32乗以上、逆順範囲、不正な区切り、重複/重なる範囲の場合、CLIは引数エラーとして実行前に拒否する。
2. If 展開結果が100000個を超える場合、CLIは大きな配列を作る前に拒否する。
3. When 引数が不正、または`--print-plan`を実行するとき、CLIは実験を実行せず、出力先を作らない。
4. The CLI shall helpと実行文書で範囲の終端・混在・重複・上限を説明する。
