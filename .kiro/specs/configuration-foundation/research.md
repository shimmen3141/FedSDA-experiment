# 設定基盤の調査と設計判断

## Summary

- 対象: `configuration-foundation`。
- 調査範囲: 既存の設定・宣言的制約を新APIへ移す拡張として、integration-focused discoveryを実施。
- 旧構成からのAPI互換は設けない。既存の計算処理とgoldenは固定した比較対象である。
- 新しい外部サービスやライブラリは不要。Python標準機能を使い、設定処理からNumPy・PyTorchを参照しない。

## Research Log

### 既存の設定解決と副作用

- 調査元: `federated_drift_experiment/experiment_spec/configuration.py`。
- `AlgorithmOptions`と`ExperimentConfiguration`は不変の型を持つが、`activated()`でグローバルconfigを書き換える。
- 新実装は機能固有の不変設定を実行層から渡す。旧`config_overrides()`の対応表を新APIの本体へ移植しない。
- `options.py`の能力・適用条件、`parameters.py`の単位の整理は活用する。旧表の名前・分割方法は引き継がない。

### 最終構成とgoldenの違い

- 正本: `docs/overview/proposed-method.md`。goldenの条件: `tests/proposed_regression_golden.json`の`definition`。
- 最終方式はResidual Adapter、全体・正解クラス別e-SR監視、Switching、joint平均勾配、FIFO再較正、前向き二分区間検証。
- goldenはさらに事前学習、optimizer、データ生成等の条件を固定している。研究の最終presetと小規模golden条件を同じpreset名へ混在させない。
- 最初の設定型だけでは全実験条件を表せない。実行層の移植前に、その実行が参照する全条件を追加・命名レビューする。

### 値の意味

- `config.py`と`clients/fedsda.py`で、FIFOは帰属保留容量、前向き検証件数は候補提案後の観測総数と確認した。
- Adapter rankは要求値を特徴次元で丸めるため、要求値と実効値を分ける。
- e-SR alphaは誤警報制御値であり、一回の検定の誤警報確率と同一視しない。
- γは再利用・候補比較・統合等で参照される。用途別の設定分離には判断式の追加調査が必要で、今回正式名を断定しない。

### 標準機能の選択

- [Python 3.13 dataclasses](https://docs.python.org/3.13/library/dataclasses.html): `frozen`は属性代入を制限するが、型注釈の値検証は実施しない。
- [Python 3.13 MappingProxyType](https://docs.python.org/3.13/library/types.html#types.MappingProxyType): 元のmappingの変更を反映する。単に包むだけでは入力からの変更を防げない。
- 不変設定はdataclassと不変値で構成する。辞書を保持する場合は呼出し元の値から切り離し、入れ子もコピー・不変化する。
- 型注釈・読取り専用mappingだけで妥当性や深い不変性が保証されるとは扱わない。

## Architecture Pattern Evaluation

| 案 | 評価 |
|---|---|
| 一つの巨大設定型 | 専用項目が混在し、選択肢の削除が難しいため採用しない |
| 機能別設定＋外側の解決・検証 | 必要な設定だけを計算部へ渡せるため採用する |
| 汎用plugin・DI framework | 今回の要求に不要で、実装や依存を増やすため採用しない |
| 外部の設定検証ライブラリ | 現在必要な型・範囲・組合せ検査は標準機能で表現できるため追加しない |

## Design Decisions

### 最小設定から段階的に接続する

- 初回は最終構成で必要な値と方式の設定・検証を扱う。小規模実行の移植は別specへ接続する。
- 本specの全受け入れ条件は、後続の一覧・preset・保存表現を含む段階まで満たしてから完了にする。
- 型だけの成功を、golden同値性の証拠として扱わない。

### 定義と実装生成を分ける

- 方式の説明・専用項目・能力・組合せ制約は宣言へ集める。具体実装のimportや生成関数はruntimeが所有する。
- 新方式の登録と実装生成の対応確認はruntime統合時の検証で行う。設定基盤だけでは実装の存在を保証しない。

### 入力の変更指定と実効条件を区別する

- 明示指定はキーの存在で判断する。値が既定値と同じでも明示指定として検査する。
- 方式変更後に無効になったpreset由来の専用値は実効条件から除く。一方、明示した専用値が非適用なら拒否する。
- 数値の暗黙変換、旧名alias、未知の設定の黙認を避ける。

## Risks & Mitigations

- グローバルconfigの見落とし: 各機能の移植時に参照項目を洗い出し、設定型と回帰ケースへ対応させる。
- 不変型の中の可変値: 呼出し元からの切離しと入れ子の変更試験で検証する。
- 二つのschemaの重複保守: 旧schemaを新実装のimport先にせず、旧名との対応はテストの比較資料に限定する。
- 承認済み名への詳細追加: revision 2の承認hashを履歴へ保存し、設計・独立レビューで追加した項目をrevision 4として確認する。

## 独立レビューの反映

gpt-6-lunaの指摘を既存コードと照合し、reset範囲、集約型の直接構築、宣言の正本、
部分型と完全型、全条件の棚卸し、FIFOとFixed-Share時間尺度の結合を具体化した。
指摘・根拠・対応と採用しなかった代替案は[luna-review.md](luna-review.md)に記録した。

## 参照した手順

fable-method、kiro-spec-design、design-principles、design-discovery-light、design-synthesis、design-review-gateを参照した。
本調査は外部仕様の新規導入ではなく、既存実装の境界整理である。
