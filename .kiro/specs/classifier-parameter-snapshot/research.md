# 調査と判断

## Summary
2026-10-06、既存PyTorch拡張、追加依存なし。旧SimpleMLP.get_params（models.py:125）はdeepcopy(state_dict())で全NN値をコピーする。旧登録（clients/base.py:322）はsnapshot→統計参照→readyを保留し、送信状態は独立した次の責務である。

## Research Log
- 新ResidualAdapterClassifierは通常bufferなしのLinear/ReLU/Sigmoid/Identity構成。共有部/adapter/分類層のnative state_dict keyと順序を新APIの名前として保持できる。
- state_dictのTensorはモデル値の借用である。dictだけをコピーするとモデルとstorageを共有するため、各値をdetach().clone()する必要がある。
- candidate-parameter-initializationはplain dictの完成snapshotを受け取り、選択/平均/独立コピーする。生成境界をここへ追加せず、一モデルのsnapshot生成をlearning/modelsへ置く。
- 旧snapshotのOrderedDict/_metadataや旧parameter keyは新形式の通常契約へ入れない。旧とのprefix対応はtest内だけで行う。
- 新通常モデルはCPU float32。snapshot生成もこの範囲へ限定し、既存候補初期化の汎用15dtype契約は変更しない。

## Design Decisions
Generalization: 送信専用名にせず一分類器の全parameter取得とし、初期化と送信準備が同じ結果を利用する。
Build vs adopt: 既存state_dictとTensorのdetach/cloneを利用。新state型/serializer/汎用モデルProtocolや互換readerを追加しない。
Simplification: 一関数・入力検証helper、IDと永続状態なし。通常モデルはbufferなしなので全parameterだけを扱う。異なる構造への拡張は明示spec変更で行う。

## 使用した手順
fable-method、導入済みcc-sddのinit/requirements（EARS/gate）・design（light discovery/synthesis/gate）・tasks・impl・review・verify-completion・validate-impl。新ライブラリと外部APIがないため追加のweb調査は不要。
