# Requirements Document

## Introduction

比較実験の利用者は、FedDriftの全体runを実行できるが、現在は設定の束を組み立てたテストから呼び、結果をメモリ上で扱っている。FedSDAと同様に、完全な設定から1 runを実行し、導出済みの指標と時系列の記録を新形式で保存・読込みできるようにする。

## Boundary Context

- 対象: 完全なFedDrift条件の検証、計測つきrunと既存指標の接続、記録の表、JSON/NPZの保存・読込み。
- 対象外: 掃引・CLI登録・既定条件のpreset、新しい診断の計算、旧成果物の読込み、アルゴリズム変更。
- 隣接: 既存の全体run・基本指標・計算量・モデル診断を使う。予測混合、候補検証、再較正の項目をFedDriftへ付加しない。

## Requirements

### Requirement 1: 完全な条件から1 runを実行する
**Objective:** As a 比較実験の利用者, I want 条件を一つに束ねて実行する, so that 実行と評価の設定が保存結果に残る。

#### Acceptance Criteria
1. When 完全なFedDrift条件を指定する, the FedDrift実験 shall 実行・参加者・指標の条件と、検出バッチ件数と集約間隔の一致を実行前に検証する。
2. If 条件の型・値・対応が不正である, the FedDrift実験 shall モデル生成と乱数消費の前に拒否する。
3. When 有効な条件で実行する, the FedDrift実験 shall 既存の計測つき全体runと指標導出を用い、条件・計測結果・全指標を返す。
4. The FedDrift実験 shall 呼出し側のPython・NumPy・CPU Torch乱数状態を維持する。

### Requirement 2: 記録を表へ変換する
**Objective:** As a 比較実験の利用者, I want 記録を時刻・client・モデルと対応した配列で得る, so that 保存後にも指標を解釈できる。

#### Acceptance Criteria
1. When 成功したrunを表へ変換する, the FedDrift実験 shall 処理した標本の予測・観測クラス・概念・モデルID・保有数と、生成した概念列全体の真の概念変更（未処理末尾を含む）、学習モデル切替、検出バッチの結果（全体runで処理したバッチのみ）、適応の記録を保存対象に含める。
2. When 成功したrunを表へ変換する, the FedDrift実験 shall モデル登録・モデルごとと対ごとのクラスタ観測・clientクロス評価・終了時保有モデルの学習量を保存対象に含める。
3. When 成功したrunを表へ変換する, the FedDrift実験 shall 通常同期後のモデル数と時刻ごとの標本処理/同期の計算量を対応させ、準備と終端の計算量を通常時刻と分けて保持する。
4. The FedDrift実験 shall 表の名前と列の名前、型、行順、空系列と欠測値の規則を安定させ、参加者状態と乱数を変更しない。

### Requirement 3: 結果を保存する
**Objective:** As a 比較実験の利用者, I want 指標と記録を保存する, so that 実験後に条件と結果を再確認できる。

#### Acceptance Criteria
1. When 存在する出力ディレクトリと単純なファイル名を指定する, the FedDrift実験 shall 形式の版・完全な条件・全指標・実行環境・表ファイル名を持つJSONと、表を持つNPZを保存する。
2. The FedDrift実験 shall 要約の非有限値をnullにし、NPZでは数値配列の欠測を維持し、pickleを必要とする配列を保存しない。
3. If ファイル名が不正、出力ディレクトリが存在しない、またはどちらかの出力が既にある, the FedDrift実験 shall 既存ファイルを変更せずに拒否する。
4. If 保存が途中で失敗する, the FedDrift実験 shall この呼出しで作った一時ファイルと未完了の成果物を除去する。成功時は表を先に置き要約を最後に置く。

### Requirement 4: 新形式を読み込む
**Objective:** As a 比較実験の利用者, I want 保存済み結果を読み込む, so that runを再実行せずに分析できる。

#### Acceptance Criteria
1. When 対応する版のFedDrift要約を指定する, the FedDrift実験 shall 同じディレクトリの対応するNPZをpickleなしで読み、要約と配列を返す。
2. If 要約がオブジェクトでない、版または手法が不一致、表ファイル名が単純なNPZ名でない、あるいは表ファイルが欠落/破損している, the FedDrift実験 shall 拒否する。
3. The FedDrift実験 shall 固定旧実装と既存goldenを変更せず、新経路だけでも実行・保存・読込みできる。
