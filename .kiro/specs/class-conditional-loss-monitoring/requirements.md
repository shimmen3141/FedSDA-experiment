# 要件: 全体・正解クラス別の損失監視

## Introduction

研究者が最終FedSDAのClassESR監視を、旧基準と同じ数値・警報・候補開始位置で独立検証できるようにする。単一損失系列のe-SRと、全体系列・正解クラス系列の固定重み混合を対象にする。

## Boundary Context

現行モデルのラベル観測後有界損失を監視する。混合予測の損失で代用しない。モデルforward・統計蓄積・baselineの推定・FIFOによる開始位置の切詰め・候補判定・警報後操作・履歴保存・新全体runは後続の責務。
baselineは呼出し側から指定し、検出器の数値安全域へ制限する。呼出し側がモデル統計からbaselineを選ぶ旧0.01下限は別の契約であり、今回の部品は統計を読む・推定しない。
初回範囲は全体/全クラスへの従来の等重み配分。独自component重み方式は含めない。
正常系列の数値は旧基準へ照合し、旧clientの不正クラスでの部分更新は引き継がず、不正入力時の状態不変を新APIの契約として検証する。

## 観測可能な入力・結果

- クラス数はbool以外のbuiltin int、2以上。保持候補数はbool以外のbuiltin int、1以上。賭け率は非空の不変な列で、各値はbool以外の有限実数、厳密に0～1の間。並び順・重複を保持する。
- 誤警報制御値alphaは既存の監視固定条件から受け取り、厳密に0～1の間。内部候補保持上限と賭け率は明示指定する。旧基準はalpha=0.001、保持上限1000、賭け率(0.05,0.1,0.2,0.4,0.8)。
- baselineと観測損失はbool以外のbuiltin int/floatの有限値で0～1。baselineの数値安全域は1e-6～1−1e-6。
- 正解クラスはbool以外のbuiltin int、0～K−1。標本位置はbool以外の非負builtin int。一つのreset区間の全体系列では初回位置を任意に指定でき、それ以降は直前位置＋1。各クラスの位置列は非連続でもよい。
- 毎標本、現在のモデルbaselineを指定する。その値は該当クラスの初出時だけ固定し、既存系列のbaselineを更新しない。全体baselineは初期化/reset時だけ変更する。
- 単一系列の候補番号はreset後1始まりの観測順。混合監視の開始位置は外部のglobal標本位置で返す。候補開始位置はFIFOの帰属位置や真の変化点とは区別する。
- 返却は混合対数e値・警報・候補開始global位置・推定変化区間長・今回更新した成分数・候補×賭け率の評価件数。初期/reset後は対数e値−∞、警報なし、候補なし、観測件数0。内部状態の観測値はコピーで取得できる。

## Requirements

### Requirement 1: 単一損失系列の監視

1. When 有界損失を一つ観測する, the Loss Monitoring System shall 各保持候補のe-processを旧基準と同じ数値・順序で更新し、その観測を開始点とする候補も追加する。
2. When 保持上限を超える, the Loss Monitoring System shall 最古候補から除き、保持した候補と賭け率の数に応じた評価件数を報告する。
3. When 候補寄与を集計する, the Loss Monitoring System shall 旧対数e値・閾値以上かどうかの警報・最大寄与候補とその年齢を再現し、寄与同率では最古候補を選ぶ。
4. When 初期化または明示resetする, the Loss Monitoring System shall 数値安全域へ制限したbaselineを固定し、候補・時刻・警報・対数e値を初期状態へ戻す。

### Requirement 2: 全体・正解クラス系列の固定重み混合

1. When ラベル観測後の現行モデル損失を受け取る, the Loss Monitoring System shall 全体成分と正解クラス成分だけを更新し、今回の成分更新数と候補評価数を報告する。
2. When クラスを初めて観測する, the Loss Monitoring System shall その時点で外部から指定されたbaselineでクラス系列を開始し、その後の指定baseline変更で既存系列を変えない。
3. When 混合対数e値を計算する, the Loss Monitoring System shall 旧等重み配分と演算順で全体と既観測クラスの最新値を混合し、未観測クラスの重みを再配分しない。
4. When 混合値を判定する, the Loss Monitoring System shall 混合値が閾値以上の場合だけ警報とし、成分単体の警報を追加でOR接続しない。

### Requirement 3: 候補位置と観測結果

1. When 混合警報が出る, the Loss Monitoring System shall 最大寄与成分の候補を選び、旧規則と同じglobal標本位置と区間長を返す。
2. While 成分寄与が同率である, the Loss Monitoring System shall 全体、今回の正解クラス、既観測の他クラスの初出順という旧優先順で選ぶ。
3. When クラス系列の候補を選ぶ, the Loss Monitoring System shall 保持されたクラス位置列からglobal位置へ対応させ、非連続なクラス出現と保持上限による切捨てを反映する。
4. When 混合警報がない, the Loss Monitoring System shall 全体成分の最大寄与候補と区間長を観測結果として返し、警報後の自動resetを行わない。

### Requirement 4: 明示条件・入力拒否・状態所有

1. The Loss Monitoring System shall alpha・監視対象・保持上限・賭け率・baseline・クラス数を明示入力から受け取り、旧グローバル設定を読まない。
2. If 型・値域・有限性・クラスID・標本順が入力契約に反する, the Loss Monitoring System shall 理由付きで拒否し、全体/クラス系列・位置・観測結果を変更しない。
3. When 混合監視をresetする, the Loss Monitoring System shall 全体を指定baselineで再開始し、全クラス系列・クラス位置・最後の標本位置・結果を破棄する。
4. The Loss Monitoring System shall 状態を実体ごとに所有し、取得した状態コピーや観測結果の変更で内部状態を変更させず、乱数・モデル・学習状態を変更しない。

### Requirement 5: 旧基準照合と責務境界

1. The Loss Monitoring System shall 低損失・上昇・同率・保持上限・baseline端点・resetの単一系列結果を旧検出器へ直接照合できる。
2. The Loss Monitoring System shall 二値/多クラス・クラス局所上昇・遅延baseline・非連続クラス位置・未観測クラス・成分優先順の混合結果を旧clientへ直接照合できる。
3. When 現行モデルの観測後損失を監視へ渡す, the Loss Monitoring System shall 予測や重みの状態を所有せず、明示損失・クラス・位置の接続を独立検証できる。
4. The Loss Monitoring System shall 旧import・互換aliasを持たず、全goldenと依存境界を検証し、この部品の完成を新FedSDA全体runの完成とは扱わない。

