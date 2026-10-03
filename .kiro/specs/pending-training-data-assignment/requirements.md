# 要件: 保留学習データの位置・FIFO管理

## Introduction

研究者が保留標本の順序・解放・警報分割を、旧最終FedSDAと同じ結果で独立検証できるようにする。

## Boundary Context

既存のFIFO容量条件と、クライアント内で連続するglobal標本位置を入力する。容量調整を伴わない追加、平時の明示超過解放、正の推定変化区間長による参照、呼出側が決めた全件消費を対象とする。
payload・帰属先モデルID・真のconcept ID・学習/損失統計・検出器/episode/候補session・終端の方針は対象外。呼出側が返却位置からデータを参照し、解放時のモデルへ帰属させる。
警報時のC+1件保持や分割後の非clearを可能にするため、追加/解放/分割参照/消費を独立した操作にする。LEGACY-002/003の改善は今回行わない。

## 入力・結果

- FIFO容量はbool以外のbuiltin int、1以上。既存の容量設定から明示的に受け取る。
- 標本位置はbool以外の非負builtin int。初回は任意位置から開始でき、それ以降は直前＋1。全件消費後も標本順は継続する。
- 推定変化区間長はbool以外のbuiltin int、1以上。0の旧slice特殊挙動は受理しない。
- 結果は観測順の変更不能な位置列。分割は前区間・変化区間・FIFO内の変化開始位置（空ならなし）。状態copyは保留位置列と最後に追加した位置（未観測ならなし）。

## Requirements

### Requirement 1: 明示追加と平時解放

1. The Pending Assignment System shall 既存容量設定と標本位置だけを入力として保留位置を所有し、旧グローバル設定やmodel/payloadを参照しない。
2. When 標本位置を追加する, the Pending Assignment System shall 観測順に末尾へ保持し、容量超過でも自動解放せず警報時の容量＋1件を保持できる。
3. When 容量超過分の解放を明示する, the Pending Assignment System shall 容量以下になるまで最古から解放し、解放位置を順に返す。容量丁度または空なら空列を返す。

### Requirement 2: 警報区間の非破壊参照

1. When 正の推定変化区間長を指定する, the Pending Assignment System shall 保留件数と区間長の小さい方を末尾の変化区間として、その前の位置列と分けて返す。
2. When 分割を参照する, the Pending Assignment System shall FIFOで切り詰めた変化区間の先頭位置を返し、FIFOより前にある検出器本来の候補位置と区別する。空FIFOは両区間空・開始なしとする。
3. While 分割結果が参照される, the Pending Assignment System shall 保留内容や最後の追加位置を変更せず、短い警報区間で呼出側が消費を見送れる。

### Requirement 3: 明示消費と状態所有

1. When 全保留位置の消費を明示する, the Pending Assignment System shall 全件を順に返して保留を空にし、最後の追加位置を維持する。空への操作は空列を返す。
2. The Pending Assignment System shall 警報・候補pending・ラウンド境界・実験終端を推測して自動消費しない。
3. The Pending Assignment System shall 実体ごとに状態を所有し、返却copy・分割結果の変更で内部状態を変えさせず、共有乱数を変更しない。

### Requirement 4: 拒否と旧基準

1. If 容量条件・位置の型/非負/連続順・区間長の契約に反する, the Pending Assignment System shall 理由付きで拒否し、全保留状態を変更しない。
2. The Pending Assignment System shall 平時解放・警報容量超過・FIFO開始位置・候補の将来検証が進行中の警報/episode重複警報での全件消費・短い区間の保持を旧clientへ直接照合できる。
3. When 監視結果の推定区間長を受ける, the Pending Assignment System shall 検出器stateやmodelを所有せず、FIFO内位置への対応をpublic APIだけで検証できる。

### Requirement 5: 完成境界

1. The Pending Assignment System shall 旧import・互換aliasを持たず、依存境界・全golden・独立起動確認を通過する。
2. The Pending Assignment System shall 完成を新FedSDA全体runや実際のモデル帰属・学習・終端flushの完成とは扱わず、発見した旧不具合/改善点を共通記録へ接続する。
