# 要求: 共有特徴抽出部への再接続

## 境界
正常に構築済みのResidualAdapterClassifierを、同じ入力寸法・隠れ層構成の既存SharedFeatureExtractorへ再接続する。
モデル一覧の共有元選択、同期parameterロード、候補採否、optimizer生成/reset、全体runは対象外。

## 受け入れ条件
- 1.1 When 適合する特徴抽出部へ接続したとき, the 分類器 shall その抽出部の同じ参照を使い、コピーや新しいNN生成を行わない。
- 1.2 When 接続を変更したとき, the 分類器 shall 概念固有adapter/分類層/出力activationの参照・値・grad・class数を保持する。
- 1.3 When 接続後にforwardしたとき, the 分類器 shall 新しい特徴抽出部を経由して概念固有部から予測する。
- 1.4 If 接続先の型/寸法/構造/CPU float32契約が不正な場合, then the 分類器 shall 接続を拒否し、元の接続と双方の値・gradを変更しない。
- 1.5 When 同じ特徴抽出部へ再度接続したとき, the 分類器 shall 既存の接続と概念固有部を保持する。
- 2.1 The 分類器 shall 接続処理でoptimizerや借用学習記録を変更せず、乱数状態を進めない。
- 2.2 When 上位が接続後に学習を準備したとき, the 分類器 shall 接続先のparameter列に対応するoptimizer状態管理器と組み合わせて利用できる。上位が接続先の共有optimizer管理器を用意・選択し、個別optimizerのresetと現在参照による新しい学習記録の作成も行う。旧接続先の共有parameter列を持つ管理器をそのまま流用しない。
