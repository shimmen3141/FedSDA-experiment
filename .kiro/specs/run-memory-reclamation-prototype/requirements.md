# Requirements Document

## Introduction

研究者はuraniumでFedSDAを多並列実行したいが、単独CLIでも5.49 GiBを使う。必要な保持データの増大は確認できず、終了後には大きな回収余地がある。学習や保存結果を変えず、実行側に閉じた同期末の回収を試作し、効果と時間コストを同じサーバで判定できるようにする。

## Boundary Context

- 対象は独立した研究CLI、専用checks、実行手順のみ。製品src・通常pytest/CI・golden・構成選択肢・既存成果物形式を変更しない。
- 既定はMNIST2/seed0/10client/5000標本/client/間隔50、回収なし→回収ありの2runを新規processで逐次実行。手元smokeの標本数だけ変更可能。
- uraniumの実行はユーザーが行い、結果を返す。本specの完了は実験手段の検証・配布まで。サーバの省メモリ効果が未確定であることを明示する。

## Requirements

### Requirement 1: 独立した再現可能な対照
1. When 計画表示を要求されたとき, the 研究CLI shall 2runの条件・順番・出力先を表示し、ディレクトリや結果を作成しない。
2. When 実行を要求されたとき, the 研究CLI shall 同じ設定の回収なし/ありを独立process・1worker・各数値thread1で逐次実行し、目的・ソース/runnerのidentity・環境を記録する。
3. If 入力不正・既存出力・対象ソース不一致・実行環境非対応なら, the 研究CLI shall 学習開始前に明示的に停止し、既存結果を上書きしない。

### Requirement 2: 必要な状態と計算の保持
1. While runが実行中のとき, the 研究CLI shall 学習データ・モデル・検証/診断記録・乱数消費・計算順・標準成果物を変更しない。
2. When 回収ありの同期が正常終了したとき, the 研究CLI shall 生存中の状態を維持したまま解放済み領域の返却を要求し、直前/直後RSSと所要時間・返却結果を記録する。
3. While どちらの対照も実行中のとき, the 研究CLI shall 追加監視thread・周期的PSS読取り・強制GC・allocator環境変数変更を行わない。

### Requirement 3: 効果と同一性の検証
1. When 2runが完了したとき, the 研究CLI shall 完全設定・全指標・全配列のキー/shape/dtype/bytes・最終サーバパラメータhashを照合する。
2. When 結果が一致したとき, the 研究CLI shall GNU timeの全体最大RSS/時間、run返却までのhigh-water、同期末での生存中回収量を分けて報告し、採否は自動決定しない。
3. If 途中失敗・未完了・コード/環境変更・照合不一致なら, the 研究CLI shall 非成功で終了し、得られた記録を保持して、成功と誤報しない。
