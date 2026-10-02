# 要求仕様: 設定・選択肢管理の基盤

## 導入

研究者が実験の方式・パラメータを理解し、選択肢を追加・削除できる設定基盤を用意する。
現在の長いmode名と散在する設定依存を、新しい正式名と検証可能な設定表現へ移行する。
本書は人間レビュー待ちの要求であり、承認済みではない。

## 境界

- 対象: 選択肢一覧、presetと明示指定の解決、値・組合せ検証、実効条件の確認・保存表現。
- 対象外: 手法の計算・学習、実験実行、CLI、ファイル保存、旧形式の変換。
- 後続への期待: runtime・CLI・掃引・成果物は同じ実効条件を使う。これらの実装は後続specで扱う。

## 要求

### Requirement 1: 設定と適用条件の把握

**目的:** 研究者が選択肢と、その方式で使用する値を理解できる。

#### 受け入れ条件

1. When 研究者がある機能の選択肢を取得する, the Configuration System shall 正式名・説明・使用するパラメータ・適用条件を返す。
2. When 登録された選択肢が追加または削除される, the Configuration System shall 選択肢一覧にその変更を反映する。
3. The Configuration System shall 数値パラメータの型・単位・範囲・既定値を説明できる。

### Requirement 2: presetと明示指定の解決

**目的:** 研究者が既定構成を選び、変更した条件を確認できる。

#### 受け入れ条件

1. When presetと明示指定を受け取る, the Configuration System shall presetを展開した後に明示指定を反映する。
2. When 最終提案のpresetが指定される, the Configuration System shall 最終提案の正本と一致する方式・固定値を設定する。
3. If 未登録のpresetが指定される, the Configuration System shall 実効設定を生成せず、対象名を含むエラーを返す。

### Requirement 3: 不正・未使用・非対応の指定

**目的:** 研究者が誤指定に気付かず別条件を実行することを防ぐ。

#### 受け入れ条件

1. If 未定義の項目または未登録の選択肢が指定される, the Configuration System shall 該当項目・値を含むエラーを返す。
2. If 数値の型または範囲が不正である, the Configuration System shall 該当項目・値・許容条件を含むエラーを返す。
3. If 選択方式で使わない専用パラメータが明示指定される, the Configuration System shall 無視せず、その項目と選択方式を含むエラーを返す。
4. If 必要な能力または組合せ条件が満たされない, the Configuration System shall 満たされていない条件を説明するエラーを返す。
5. If 旧mode名または旧option名が指定される, the Configuration System shall 新形式として受理せず、未定義の指定として拒否する。

### Requirement 4: 実効設定の確認と保存表現

**目的:** 研究者が解決後の条件を確認し、その条件を再び使用できる。

#### 受け入れ条件

1. When 設定解決に成功する, the Configuration System shall 選択方式に適用する値だけを含む実効設定を返す。
2. When 実効設定を保存表現へ変換する, the Configuration System shall 方式・適用値・設定schemaのversionを保持する。
3. When 対応するversionの保存表現を読み込む, the Configuration System shall 保存前と等しい実効設定を復元する。
4. If 未対応のversionを読み込む, the Configuration System shall 自動変換せず、未対応versionを含むエラーを返す。
5. The Configuration System shall 入力の設定・presetを変更せず、実効設定を生成する。

### Requirement 5: 削除・登録漏れの検出

**目的:** 開発者が削除した選択肢への参照や、不完全な登録を発見できる。

#### 受け入れ条件

1. If presetが存在しない選択肢またはパラメータを参照する, the Configuration System shall preset名と参照先を含む整合性エラーを返す。
2. If 同じ正式名の選択肢が重複登録される, the Configuration System shall 重複名を含む整合性エラーを返す。
3. If 定義の既定値がその定義の許容条件を満たさない, the Configuration System shall 対象項目を含む整合性エラーを返す。
