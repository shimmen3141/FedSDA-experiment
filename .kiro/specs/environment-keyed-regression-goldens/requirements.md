# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者と、このリポジトリをcloneして使う人。

現在の状況: 最終構成のgoldenは、Windows用（tests/proposed_regression_golden.json。固定）と、Linux用（WSL Ubuntuで作った1つ）がある。testは、OSの名前で、どちらかを選ぶ。研究室サーバ（Linux、Python 3.10.16、NumPy 2.2.6、torch 2.12.1+cpu）で実行したところ（2026-10-11、ユーザー）、新実装は、同じprocessの中の旧実装と3ケースとも一致したが、旧実装のsine2の結果が、WSL（Python 3.14.4、NumPy 2.4.6）と違った（sea2とmnist2は同じ）。OSの名前だけでは、goldenを選べない。

ユーザーの決定（2026-10-11）: Linux用のgoldenは、環境ごとに持つ。specのたびに、研究室サーバでユーザーが実行することは、避ける。goldenの追加や削除が容易で、cloneした人が、分かりにくい設定・操作を必要としない形にする。

## Introduction

最終構成のgoldenを、実行環境ごとのファイルとして1つのディレクトリに置き、testが、実行環境に合うgoldenを自動で選ぶようにする。

## Boundary Context

- **In scope**: 環境ごとのgoldenの置き場所と名前。実行環境に合うgoldenの選択。goldenを足す道具。goldenのない環境での扱い。既存のLinux用のgolden（WSL）の移動。文書。
- **Out of scope**: Windows用のgolden（tests/proposed_regression_golden.json）と既存の回帰test（tests/test_proposed_regression.py）の変更・移動。旧の代表11ケースのgolden。研究室サーバ用のgoldenの作成そのもの（ユーザーが、1回だけ実行する。手順を書く）。sourceの変更。
- **Adjacent expectations**: Windowsの基準環境とWSLでの照合の結果は、変えない。

## Requirements

### Requirement 1: 環境ごとのgolden

#### Acceptance Criteria

1. The 環境ごとのgolden shall 1つのディレクトリに、環境ごとに1ファイルで置かれ、ファイル名は、そのgoldenの環境の記録（OS、機種、Python・NumPy・torchの版）から決まる。
2. The 固定のWindows用のgolden shall 場所を変えずに、同じ選択の対象になる。
3. When goldenを削除したいとき, the 利用者 shall そのファイルを消すだけでよい（ほかの設定を、変える必要がない）。
4. The goldenの集まり shall 環境の記録が、互いに重ならない。条件の定義は、どれも、既存の回帰testの現在の定義と一致する。形（キー、33指標、31の離散列、イベント件数）は、どれも同じ。

### Requirement 2: 実行環境に合うgoldenの選択

#### Acceptance Criteria

1. When testを実行したとき, the 選択 shall 実行環境の記録（既存の回帰testの`environment()`）と、完全に一致する記録を持つgoldenを選ぶ。設定や引数は、要らない。
2. While 一致するgoldenがあるとき, the 照合 shall 旧実装の3ケースと、新実装の3ケースを、そのgoldenと照合する（基準は、既存の回帰testと同じ）。
3. While 一致するgoldenがないとき, the goldenとの照合 shall 実行環境の記録と、goldenを足す方法を書いた理由で、skipする（失敗にしない）。同じprocessの中の、新実装と旧実装の照合は、どの環境でも行う。
4. While 実行環境が、固定のWindows用のgoldenの環境のとき, the 旧実装の照合 shall 既存の回帰testに任せる（同じ計算を、2回しない）。新実装は、固定のWindows用のgoldenと照合する。

### Requirement 3: goldenを足す

#### Acceptance Criteria

1. When goldenのない環境で、作成のコマンドを実行したとき, the 道具 shall 旧実装の3ケースを実行して、その環境のgoldenのファイルを、決まった名前で書く。
2. If 実行環境に、すでにgoldenがあるとき, the 道具 shall 上書きを、明示の指定（`--overwrite`）がなければ拒否する。固定のWindows用のgoldenの環境では、常に拒否する（既存の回帰testの`--update`で更新する）。
3. The 作成 shall 既存の回帰testの計算（`compute_all`）を、そのまま使う。

### Requirement 4: 手順の記録

#### Acceptance Criteria

1. The 基準環境の文書 shall goldenの置き場所、選び方、足し方、消し方、goldenのない環境での挙動、goldenを作り直す条件（意図したアルゴリズムの変更を承認したとき。ふだんのspecでは、作り直さない）を書く。
