# 要件: モデル別学習・割当件数

3つのモデル別計数を独立した初出順で所有する。計数の入力は上位で確定済み。学習/抽出/回数計画・全体compute counter差分計算・採番/現在帰属ID・標本/NN/統計・全登録/通信は含めない。概念IDは診断専用である。

## Requirement 1: 計数とsnapshot

1. When 完了済み学習量が渡される, the Counts Store shall モデルへ延べ学習標本数と個別parameter update step数の非負増分を加算する。0でも両計数のIDを登録し、それぞれ初出順を保持する。
2. When 帰属済み標本の真概念が渡される, the Counts Store shall 既知conceptのモデル別件数へ1加算し、Noneでは項目を生成しない。concept初出順を保ち、学習標本数/step数を変えない。
3. The Counts Store shall 3辞書の独立snapshotと一モデルの概念件数コピーを返す。欠落conceptの読み取りは空dictを返しownerへ項目を作らない。取得結果の変更と後続更新は互いの辞書構造へ伝播しない。

## Requirement 2: 加算移管と再編

1. When 異なる元/先IDへの単一移管を要求される, the Counts Store shall 元が存在する計数だけ元を除き先へ加算する。先既存は位置維持、新先は末尾、元欠落はno-opで先を生成しない。概念は先順を保って共通conceptへ加算し新conceptを元順で追加する。
2. When 一回ID対応表を適用する, the Counts Store shall 各辞書を独立に元順で再編し、先初出順で合計する。conceptも元/内部順に加算し、未対応ID・0/空concept項目を保持し、連鎖や循環を再帰適用しない。
3. When 単一移管の元と先が同一, the Counts Store shall 事前に理由を示して拒否し、全計数を保持する。旧不正通知の破損を再現する経路を新APIへ持ち込まない。

## Requirement 3: 入力・依存・接続

1. The Counts Store shall 全モデルID、既知concept ID、増分をbool/派生型以外のbuiltin intで検査する。増分は非負、conceptはNoneも許す。対応表はexact dictと全key値を状態変更前に検査する。欠落元/Noneの場合も該当入力を検査し、不正項目と理由を示して原子的に拒否する。
2. The Counts Store shall 計数以外のowner・共有RNG/数値環境を変更せず、数値ライブラリ・旧実装・学習実行/回数設定・runtime/global設定へ依存しない。
3. The Counts Store shall 正常な旧計数/移管/一回再編と、実NNの共同学習後の計数→正式ID移管→後続学習の明示接続で、計数/順序と全損失・parameter/grad/optimizer/Randomを一致させる。
