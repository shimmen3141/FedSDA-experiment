# Research & Design Decisions

## Summary

- **Feature**: `fedsda-experiment-run-artifacts`
- **Discovery Scope**: Extension（既存の実行と導出をつなぎ、保存を足す）
- **Key Findings**:
  - 「設定→計測つきの全体run→指標の導出」は、既存の3つの関数の呼出しで済む。
  - 旧の31の配列は、新の記録（標本ごとの予測、適応、判定、登録、クラスタリングの観測）から、testの中の対応（`derive_legacy_trace_arrays`）で作れている。保存する表は、この対応の、新の側の項目である。
  - 旧のラウンドごとのモデル数は、同期の後の値。新の記録からは、グローバルモデル数＝初期の1＋登録の累計−統合で吸収された数の累計、クラスタリング待ち＝直前のクラスタリングの後に登録されたモデルの数（クラスタリングを行ったラウンドは0）、clientごとの保有モデル数＝次のラウンドの最初の標本の、処理の前の保有モデル数（最後のラウンドは、終端の状態）で求まる見込み（実旧と照合して確かめる）。
  - 新の適応の結果種別は、旧の`action`より細かい（複数の結果種別が、旧の1つの`action`に対応する）。候補の判定の理由は、1対1。

## Design Decisions

### Decision: 保存の単位は、1 runにつき、要約（JSON）と、記録の表（NPZ）の2ファイル

- **Alternatives Considered**: (1)旧と同じく、NPZの1ファイルに、条件も指標も入れる——条件と指標は、入れ子の値で、配列に向かない（旧は、指標をJSONの文字列にして、NPZへ入れている）。(2)指標を、掃引のCSVの行としてだけ保存する——1 runの結果が、掃引の道具に依存する。
- **Selected Approach**: 要約は、人が読めて、道具で集計しやすいJSON。表は、配列をそのまま持てるNPZ。掃引の集計（CSV）は、要約のJSONから作る（(A3)）。
- **Rationale**: 1 runの結果が、それだけで完結する。

### Decision: 表の配列の名前は、`表の名前.列の名前`

- **Selected Approach**: 記録の種類ごとに表を決め、列は、新の記録のfieldの名前にする（例: `adaptation.adaptation_outcome`、`sample_prediction.combined_prediction_is_correct`）。事象の表（概念の変更、適応ほか）は、行ごとに`client_id`を持つ1次元の列。標本ごとの表は、client×標本の2次元。ラウンドごとの表は、ラウンド（×client）。
- **Rationale**: 列の意味が、名前から分かる。表ごとに、行の数がそろう（検査できる）。

### Decision: 適応の表は、細かい結果種別と、粗い区分の、両方の列を持つ

- **Selected Approach**: `adaptation.adaptation_outcome`（新の結果種別）に加えて、`adaptation.adaptation_action`（維持・再利用・作成・作成の棄却・作成の保留・サーバの統合、の区分）を持つ。区分は、結果種別から決まる。
- **Rationale**: 旧の結果は、粗い区分しか持たない。旧の結果を変換した表と、新の表が、同じ列（`adaptation_action`）を持てる。区分の名前は、新の語で付ける（旧の値との対応は、(A2-変換)で持つ）。

### Decision: 初期モデルの登録ラウンドは、−1で表す

- **Selected Approach**: `model_registration.registered_round_index`は整数の配列なので、初期モデル（登録のラウンドがない）を−1にする。
- **Rationale**: 整数の配列に、欠損を入れられない。旧も−1。

### Decision: 上書きしない。書きかけを残さない

- **Selected Approach**: 保存先に、どちらかのファイルがあれば、拒否する。一時の名前へ書いてから、名前を変える。失敗したら、一時のファイルを消す。
- **Rationale**: 長時間の実験の結果を、誤って消さない。並列の実行で、壊れたファイルを残さない。

### Synthesis

- **Build vs. Adopt**: 条件と指標の辞書は、既存の`convert_settings_to_plain_mapping`を使う（指標の非有限の値を`null`にする指定を足す）。
- **Simplification**: 形式の版は、整数1つ。来歴（commit、実行のコマンド）と、保存先の名前の規則は、(A3)。

## Risks & Mitigations

- 表が、旧の31の配列を作るのに足りない — 保存して読み込んだ表だけから、旧の31の配列を作り、実旧が保存した配列と照合する。
- ラウンドごとのモデル数の求め方が、旧と違う — goldenの3ケースと、登録・統合を通る小さい条件で、実旧の値と照合する。
