# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [ ] 1. clientの設定の束
- [ ] 1.1 置き場所のない値の型と、clientの設定の束を実装する
  - 数値と文字列の値を、既存の検査の仕組みで宣言する型にまとめ、機能別の設定・賭け率と合わせた束にする。型、範囲、賭け率、2組の等値（Fixed-Shareの時間尺度と保留の容量、最小改善量と早期終了の最小改善量）を、生成時に確かめる。
  - 新しいmoduleの依存の許可集合を、既存と同じ形で登録する。
  - 完了: 束の単独のtest（各値の境界、別の型、賭け率の不正、等値の違反、frozenを回避した不正の拒否）と、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunClientScalarSettings、FedsdaRunClientSettings_
  - _Requirements: 5.1, 5.2, 5.3, 6.3_

- [ ] 2. clientの組立てと操作
- [ ] 2.1 clientの組立てを、実旧の生成直後の状態との対照つきで実装する
  - 実`_pretrain_initial_model`と実`__init__`で実旧の最終構成のclientを作り、新側を同じ値の初期モデルから組み立てる対照testの土台を先に作る（旧の設定の差し替えを1箇所にまとめる）。
  - 組立ての引数の検査、初期モデルの写し、全ownerの生成、ownerの記録とclientの生成を実装する。
  - 完了: 生成直後の全状態が実旧と一致するtest（2値・多クラス）、組立ての拒否のtest（各不正で、渡した初期モデル・optimizerの状態・乱数が変わらない）、写しの独立と乱数を使わないことのtestが成功する。
  - _Boundary: assemble_fedsda_run_client、FedsdaRunClientOwners_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.4, 5.5_
- [ ] 2.2 clientの5つの操作を実装する
  - 観測標本の変換と標本1件の処理の呼出し、ラウンド境界での保留中の学習、登録できるモデルの有無、送信待ちの進行、終端での未完了の候補検証の回収（概念IDの保持の解除を含む）を実装する。新しいmoduleの依存の許可集合を登録する。
  - 完了: 対照testが、2値・多クラス、複数の標本列で、標本ごと・ラウンド境界ごと・終端の後に、全状態と乱数の一致を示し、要求4.2の経路をすべて通ったことを確かめるtestが成功する。操作の入力の拒否のtest、概念IDなしのtest、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunClient_
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 5.6, 6.3_

- [ ] 3. 実行の枠への接続
- [ ] 3.1 組み立てたclientを、実行の枠の参加者として動かす
  - 組み立てたclient 2つと、何もしないサーバの代役で参加者を作り、参加者の検査と区間の進行を実行するtestを追加する（sourceの変更が要らないことを確かめる）。
  - 完了: 参加者の検査が通り、区間の進行が最後まで実行され、実行の記録の段の列が契約の順で、各clientの予測の記録の件数が処理した標本数と同じである。
  - _Depends: 2.2_
  - _Boundary: FedsdaRunClient（実行の枠との適合）_
  - _Requirements: 6.1_
- [ ] 3.2 共用のfresh process scriptへ、clientを組み立てて実行の枠で進める流れを足す
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、clientを組み立て、区間の進行で標本列を最後まで進め、予測の記録の件数と、警報が1回以上起きたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.2_

- [ ] 4. 検証
- [ ] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.2_
  - _Requirements: 4.1, 6.1, 6.2, 6.3_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
