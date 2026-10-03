# 要件: 重み付き分類予測

## Introduction

研究者が最終Residual Adapter＋Switchingの混合予測と観測後損失を、旧基準と同じ数値で独立検証できるようにする。
今回はすでに得られたモデル出力の確率化、全モデルの重み正規化、混合とクラス判定、観測後のモデル別平均有界損失を扱う。

## Boundary Context

- 既存CPU float32基準の二値・多クラス分類を対象とし、複数標本では同じ操作順の平均損失を扱う。
- 予測前に正解ラベル・真の概念を受け取らない。モデルforward・学習・候補・監視・FIFO・診断・保存は後続範囲。
- 重み状態は上流specに属する。混合と観測後更新へ同じ正規化後の重みを渡す。
- active集合選択、AdaHedge/meta、GPU/別精度対応は含めない。

## 観測可能な入出力契約

- クラス数Kはboolを除く整数でK≥2。標本数Nは正。同じ呼出しの全モデルでNが一致する。
- モデル出力はCPU float32のdense表。二値は[N,1]の有限なクラス1確率（0～1）、多クラスは[N,K]の有限logit。
- モデル別確率も同じ形・配置・精度。各値は0～1、多クラスの各行の総和は1との差1e-6以内。検査で値を補正しない。
- クラス判定に渡す混合後スコアは同じ形・配置・精度の有限表。逐次float32加算による誤差を再拒否しないため、判定側では値域・行総和を検査せず、その値のまま閾値比較/最大値選択する。確率としての値域・行総和の検査は混合前のモデル別確率に対して行い、混合結果を補正しない。
- 予測クラスの返却は[N,1]、CPU float32の整数クラス値。これは旧返却表現を保持するためで、損失計算用ラベルの型とは区別する。
- モデルIDはboolを除くbuiltin intで、負ID可。Mappingは非空。確率と重みのID集合は完全一致する。
- 重みはboolを除くbuiltin int/floatの有限値で0～1、総和は1との差1e-12以内。正規化ではID昇順の通常の加算と個別除算を使う。正規化後の値を混合中に再正規化しない。
- 観測ラベルは同じCPU上の[N]または[N,1]、int32/int64/float32。整数クラス値0～K-1だけを受理し、小数・bool・非有限・件数不一致を拒否する。
- 平均損失はfloat32の標本平均を旧経路の演算順で求めてPython floatへ返す。入力モデル別確率を使い、混合確率からモデル損失を代用しない。

## Requirements

### Requirement 1: モデル出力の確率化

**Objective:** 研究者が出力の意味を混同せず、ラベル観測前の確率を得る。

1. When 二値モデルの出力を受け取る, the Classification Prediction System shall sigmoid済みのクラス1確率をそのまま使い、再度sigmoidを適用しない。
2. When 多クラスモデルのlogitを受け取る, the Classification Prediction System shall モデルごとにクラス軸のsoftmaxで確率へ変換してから混合へ渡す。
3. The Classification Prediction System shall モデルを再実行せず、正解ラベルや真の概念を受け取らずに確率化する。
4. If 出力の形・クラス数・精度・配置・値が入力契約を満たさない, the Classification Prediction System shall 理由付きで拒否し、入力を変更しない。

### Requirement 2: 重み正規化と混合

**Objective:** 研究者が全モデル混合と観測後更新に同じ重みを使える。

1. When 予測前の全モデル重みを受け取る, the Classification Prediction System shall 旧経路と同じ順序の正規化済み重みを独立した値として返す。
2. When モデル別確率と正規化済み重みを受け取る, the Classification Prediction System shall ID昇順の積・逐次加算で旧基準と同じ混合確率を返し、混合時には重みを再正規化しない。
3. While モデルが一つまたは一部の重みがゼロである, the Classification Prediction System shall 同じ混合規則を適用する。
4. If 集合が空、IDが不正、重みの値域・総和またはモデルID対応が不正である, the Classification Prediction System shall 入力を変更せずに拒否する。

### Requirement 3: クラス予測

**Objective:** 研究者が旧基準と同じ閾値・同率規則でクラスを予測する。

1. When 二値混合スコアから予測する, the Classification Prediction System shall 0.5より大きい場合だけクラス1とし、0.5ちょうどはクラス0とする。
2. When 多クラス混合スコアから予測する, the Classification Prediction System shall 最大スコアのクラスを選び、同率なら最小クラス番号を選ぶ。
3. The Classification Prediction System shall 標本ごとのクラス予測を旧基準と同じ形・精度で返し、正解ラベルの変更で予測結果を変えない。
4. If クラス判定用スコアが入力契約に反する, the Classification Prediction System shall 入力を変更せずに拒否する。

### Requirement 4: ラベル観測後のモデル別損失

**Objective:** 研究者が予測に用いた同じモデル別確率から、重み更新へ損失を渡せる。

1. When 二値分類の正解ラベルを観測する, the Classification Prediction System shall 各モデルのクラス1確率とラベルの絶対差を標本順で平均する。
2. When 多クラス分類の正解ラベルを観測する, the Classification Prediction System shall 各モデルの正解クラス確率を1から引いて標本順で平均する。
3. The Classification Prediction System shall モデル別損失を旧基準と同じ数値で重み状態部品へ渡せる形で返し、モデルを再実行しない。
4. If ラベルの型・値・件数・形・配置が不正である, the Classification Prediction System shall 拒否し、確率・ラベル・外部の重み状態を変更しない。

### Requirement 5: 独立性・基準照合

**Objective:** 研究者が数値差と責務越境を局所化できる。

1. The Classification Prediction System shall 入力・グローバル設定・乱数・学習状態を変更せず、返却値を入力から独立させ、学習用の勾配履歴を返さない。
2. The Classification Prediction System shall 二値閾値・多クラス同率・単一/複数モデル・負ID・複数標本・重み正規化と損失を旧経路へ直接照合できる。
3. When 予測前取得・正規化・混合・ラベル観測・重み更新を順に接続する, the Classification Prediction System shall 正規化後の同じ重みを混合と更新へ渡し、旧経路と同じ予測・損失・次回重みを再現する。
4. The Classification Prediction System shall 旧moduleや互換名を新実装へ導入せず、この数値部品の完成を新FedSDA全体runの完成とは扱わない。

