# Requirements Document

## Project Description (Input)

FedSDAとFedDriftを比較する研究者向け。FedDriftの全体runと基本指標の移植は完了したが、FedSDAで一新したモデル計算の定義へFedDriftがまだ接続されていない。既存の共通計測を使って、FedDriftの全体runと時刻ごとの計数を取得する。

## Boundary Context

- 対象: モデル計算の実測・逆伝播の見積り、初期準備と本実行の分離、時刻ごとの標本処理/同期/終端の内訳。
- 対象外: サーバのパラメータ算術、保有モデル数と正規化、損失監視用途別の計数、追加診断、成果物/CLI。これらは後続spec。
- 隣接契約: 既存の全体run、失敗stage、基本指標、局所乱数と呼出し元の大域乱数を維持する。固定旧実装とgoldenを変更しない。

## Requirements

### Requirement 1: 共通定義による全体runの計測

1. When 計測つきrunを実行したとき, the FedDrift実行 shall 結果と実参加者と共通ModelComputationCountsを返す。
2. The 計測 shall FedSDAと同じ定義で全結合層の順伝播MAC、勾配有効時の逆伝播見積り、部品への入力標本数、optimizer更新数を数える。バイアス・活性化・損失・optimizer内部算術は含めない。
3. The 計測 shall 独立モデルの隠れ層と分類層をすべて概念固有部として数え、共有部は0とする。
4. The 本実行の計数 shall 初期モデル準備の計算を除外し、準備計数を別に返す。

### Requirement 2: 時刻ごとの内訳

1. When 通常同期が終わったとき, the 計測 shall 時刻順に標本処理と同期のモデル計数を別々に返す。
2. The 標本処理の計数 shall 全clientの予測と検出バッチ評価を含み、同期の計数はクロス評価とR回のローカル学習を含む。
3. The 終端計数 shall 最後の通常同期後の終端処理を含み、全時刻の2区分と終端の合計は本実行計数と各fieldで一致する。
4. If 通常同期が0回のとき, the 計測 shall 時刻列を空にし、本実行の全計数を終端区分に含める。未処理tailを学習・予測しない。

### Requirement 3: 観測の無干渉と検証

1. The 計測つきrun shall 通常runと同じ結果・全参加者状態・基本指標・局所乱数消費を持ち、大域Python/NumPy/CPU torch乱数を変更しない。
2. If runの検査または操作が失敗したとき, the 計測 shall 既存RunExecutionErrorのstageと原因を維持し、hookを外す。成功時もhookを残さない。
3. The 検証 shall 実旧入口との全状態照合、独立の小モデルMAC算式、複数dataset/optimizer/R/ドリフト/ゼロ同期、繰返し実行、失敗後の正常実行で計数と無干渉を確かめる。
4. The 新実行経路 shall 旧importに依存せず、共用fresh processとexact依存境界で検証できる。
