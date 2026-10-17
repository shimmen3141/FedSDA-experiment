# 実験掃引の再開補助

## Project Description
研究者がworker数を変えて掃引を再開するとき、出力先を探し直さずに同条件の過去掃引と進捗を選べるようにする。保存中断も安全に復旧する。NPZ、run JSON、一覧CSV、README、sweep_recordの既存形式・版・数値は変えない。追加の補助ファイルのみ許可された。

## Requirements
### 1. 過去掃引の選択
- 1.1 When 出力先を省略して対話実行する, the system shall results配下の同じ全run設定・動作基準版の掃引を進捗（67/240 run）とパスで表示し、再開・新規・キャンセルを選ばせる。
- 1.2 The system shall worker数、引数順序、目的、ラベル、studyを実験条件の一致判定から除き、完全なrun設定の集合を比較する。
- 1.3 When 完了済み掃引を選ぶ, the system shall 成功終了し、実験・README・表・実行記録を追加しない。
- 1.4 If 非対話入力で同条件が見つかる, the system shall 入力を待たず明示的な再開先または新規指定を案内して終了する。明示的出力先と新規指定は質問せず進める。
- 1.5 When print-planを指定する, the system shall 質問・作成・復旧せず候補と状態を読み取り専用で表示する。
### 2. 進捗と照合
- 2.1 The system shall 完全に読める同設定・同基準版のJSON/NPZ組だけを完了数に含め、未完了を分母から除かない。
- 2.2 The system shall 保存中断・破損・条件不一致を補足表示し、旧READMEから条件を確認できる中断掃引も候補に含める。
- 2.3 If 条件や基準版を確定できない, the system shall 自動再開候補から除き、成果物を変更しない。
### 3. 安全な再実行
- 3.1 The system shall 新規の各run開始前に完全設定・基準版・コード来歴・数値環境を補助記録へ原子的に保存する。
- 3.2 If JSONなしNPZありで、開始記録と現在の設定・clean commit・基準版・数値環境が一致する, the system shall NPZを補助領域へ退避してrunを最初から再実行する。
- 3.3 If 開始記録がない・不正・不一致、またはその他の破損がある, the system shall 元ファイルを保持し、要確認とする。正常な完了組を上書きしない。
- 3.4 While 同じ出力先を別processが実行中, the system shall 二重実行・復旧・記録追加を拒否する。正常終了・Ctrl+C・process終了後に再開できる。
### 4. 不変条件
- 4.1 The system shall 既存の成果物形式・実験数値・乱数・golden・動作基準版を変更しない。
- 4.2 The system shall run途中の学習状態を再開せず、未完了runだけを最初から計算する。
- 4.3 The system shall 既存補助記録と矛盾する計画や読み書き不能な補助記録を黙って置換しない。
