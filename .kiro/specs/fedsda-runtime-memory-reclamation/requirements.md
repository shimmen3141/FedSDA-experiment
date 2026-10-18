# Requirements Document

## Introduction

uraniumでFedSDAの並列実験を行う研究者は、解放済み領域の保持で1runの最大RSSが数GiB増える問題を持つ。同期末回収の本規模対照では全保存値を維持して最大RSSを81.54%削減できた。通常掃引からも同じ回収を使い、既存の実験コマンドと成果物を維持する。

## Boundary Context

- In scope: FedSDA全体runの実行資源管理。通常掃引の逐次・worker内実行、直接の全体runにも同じ動作を適用する。
- Out of scope: FedDrift/FedAvgへの適用、学習・評価・同期アルゴリズム、標本表現変更、GC、allocator環境変数変更、新しいrun設定・CLI選択肢・保存項目。
- Adjacent expectations: 未対応環境でも従来の実験を実行できる。今回の削減率を全環境・全条件へ保証しない。

## Requirements

### Requirement 1: 正常同期末の自動回収
**Objective:** 研究者として、既存コマンドのまま生存状態を維持して解放済みメモリを返したい。

#### Acceptance Criteria
1. When 対応するLinux/glibc環境でFedSDAの同期が正常終了したとき、実行部は各同期につき1回だけ解放済み領域の回収を要求する。
2. If 同期が例外で終了した場合、実行部はその同期の回収を行わず、元例外を伝える。
3. While 同期末回収を行う間、実行部はモデル・標本の生存参照、乱数状態、学習・予測・評価の順序と計算量記録を変更しない。
4. When 通常掃引のworkerまたは直接の全体runで実行するとき、実行部は同じ回収規則を適用する。

### Requirement 2: 環境依存の隔離
**Objective:** 研究者として、既存のWindows環境を含め、追加の設定なしで実験を継続したい。

#### Acceptance Criteria
1. If Linux/glibc以外、または回収機能をロードできない場合、実行部は回収を省略して従来どおり処理を続ける。
2. When 回収要求で返却対象がない場合、実行部は実験失敗として扱わない。
3. The 実行部 shall 環境・回収機能の準備をprocess内で再利用し、追加thread・監視・明示GCを作らない。

### Requirement 3: 結果同一性と通常経路の検証
**Objective:** 研究者として、省メモリ化だけが変わり、過去の研究結果を維持したい。

#### Acceptance Criteria
1. When 同一環境・条件で回収有無を比較したとき、実行部は完全設定・全指標・全保存配列dtype/shape/bytes・最終サーバパラメータを一致させる。
2. The 実行部 shall 既存のCLI・実験設定・JSON/NPZ/README形式と実行イベント列を維持する。
3. When 完了と判定するとき、開発者は環境分岐・同期成否のテスト、通常Linux経路の実NN対照と複数worker接続、固定旧/最終goldenを含む全pytest、共用fresh smoke、品質検査の証拠を残す。
