# 要求: モデル別学習標本の保持

## 目的と境界
モデルに割り当て済みの観測標本列を保持し、既存のbatch抽出へ渡す。
統計・概念ラベル・評価標本・モデル/optimizer・通信・候補管理・全体runは扱わない。
一時モデルの正式登録時に行うpop/上書きは別責務で、今回のID対応はサーバ統合の一回対応のみ。

## 1. 追加と順序付き参照
- 1.1 When 初期化したとき, the 保持器 shall 空のモデル別標本列を返す。
- 1.2 When 標本列を追加したとき, the 保持器 shall モデルの初出順・標本の追加順・同じ標本の重複を保存する。負の一時モデルIDも許容する。
- 1.3 When 空の標本列を追加したとき, the 保持器 shall 未登録モデルについても空の列を作成する。読み出しは未登録モデルを作成しない。
- 1.4 When snapshotを取得したとき, the 保持器 shall 以降の追加・ID対応から独立したtupleの構造を返し、標本recordとTensorは参照を借用する。Tensorのdeep copyや完全なpayload不変性を保証しない。
- 1.5 If 不正な追加入力を受けた場合, then the 保持器 shall モデルID（boolを除くbuiltin int）、exact tuple、全要素のexact ObservedTrainingSampleを事前検証し、例外前の状態を保持する。Tensorの内容検証は既存samplerへ委ねる。

## 2. 一回のモデルID対応
- 2.1 When ID対応を適用したとき, the 保持器 shall 各元IDに一回だけ対応表を引き、未指定IDを保持する。対応を推移的に辿らない。
- 2.2 When 複数の元IDが同じ宛先へ対応したとき, the 保持器 shall 元モデルの保持順で列を連結し、宛先の初出順・空列・重複標本を保存する。容量制限や抽出を行わない。
- 2.3 If 不正な対応表を受けた場合, then the 保持器 shall exact dictと使用しない項目も含む全キー/値のbuiltin intを事前検証し、例外前の状態を保持する。

## 3. 抽出への提供と責務分離
- 3.1 When snapshotを学習batch抽出へ渡すとき, the 保持器 shall 既存のModelTrainingSampleCollection列として利用でき、保持順と標本位置を保存する。
- 3.2 The 保持器 shall 標本構造の保持・追加・一回ID対応だけを担い、乱数状態やモデル/optimizer/統計を変更しない。
