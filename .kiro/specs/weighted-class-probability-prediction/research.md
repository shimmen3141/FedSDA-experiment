# 調査と設計判断

## Summary

既存経路を移植するlight discovery。fable-method・kiro-spec-init/requirements/designの境界・EARS・設計統合の規則を適用し、主担当が以下の判断を統合した。

## Research Log

- `federated_drift_experiment/models.py:15–62`とSharedBackbone/ResidualAdapterの出力: 二値はsigmoid済[N,1]、多クラスはlogit[N,K]。
- `clients/fedsda.py:1531–1588`: 全poolでも通常sumによる重み正規化、モデルID反復順のTensor×Python floatとPython sum(start=0)、二値>0.5、多クラスargmax、float32クラス表、absまたは1−正解確率のmean。
- `clients/fedsda.py:1597–1662,1763,2040`: 通常の最終構成は昇順。多クラスsoftmaxを混合前に行い、混合に使った正規化後重みをSwitchingの更新にも渡す。
- `_AdaHedgeRoutingFedSDAClientMixin`のstatic関数をテスト側oracleとして直接呼べる。client・モデル・global設定変更なしで照合できる。
- `clients/shared_backbone.py:170–185`: 特徴一回・各head一回のforward。この所有は後続モデルspec。今回の数値関数へモデルやxを渡さない。
- `tests/test_shared_backbone.py:364`等にforward回数の既存検証がある。今回の分離で追加forwardを導入しない。

## Architecture Pattern Evaluation

| 案 | 判断 |
|---|---|
| client予測全体の移植 | 診断・重み状態・モデルforwardを再結合するため見送る |
| 汎用predictor interface/結果snapshot型 | 単一数値処理には過剰なため見送る |
| learning/predictionのstateless関数 | 採用。torch計算の所在と下向き依存を明確にする |

## Design Decisions

- Build vs Adopt: 既存torch 2.12.1+cpuとstdlibだけを使い、旧演算を新APIへ移す。新依存は不要。
- Simplification: 確率化・重み正規化・混合・クラス判定・モデル損失の5関数。永続状態・framework・旧aliasを作らない。
- 重み正規化はfsumで検証した後、演算本体ではID昇順の通常sumと個別除算。混合時に二度目の正規化をしない。
- CPU float32 denseの既存基準を初期範囲とし、未対応dtype/deviceを黙ってcastしない。
- モデル別確率は厳密0～1と多クラス行総和1e-6。クラス判定は有限スコアだけを比較する。Lunaが行誤差と混合誤差の伝播による自己拒否を指摘したため、混合結果に再検査・clip・正規化を入れない。
- 主担当probe: 全値1・10モデル・各重み0.1を旧順で混合すると1.0000001192092896。入力確率検査とクラス判定スコア検査を区別する根拠。
- 入力へinplace操作しない。返却tensorは入力とstorageを共有せず、no_grad/独立copyで予測用結果を返す。

## Risks & Mitigations

重みの二重正規化、混合のvector化、ラベル再forward、浮動誤差のclipは直接oracleと前後接続テストで検出する。旧production・goldenの差分なしと新数値部品の同値性を別々に検証する。
