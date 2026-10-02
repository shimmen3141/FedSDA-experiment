# 単一run specの調査・レビュー記録

## 2026-10-03: 要件作成前の確認

- 直前の初回設定基盤: 14実行単位完了、統合1372件成功、golden変更なし。全configuration-foundation要求の完成ではない。
- ユーザーの「次の段階へ進む」指示と既存roadmapに従い、小規模実行の契約定義へ進んだ。
- 実行とデータ供給を新spec、既存設定への不足追加を隣接する設定契約として区別した。
- 学習・検出・候補・統合の全移植をこのspecへ含めず、データと処理順の検証可能な単位に限定する案を用意した。

## 旧SINEケースの実行時調査

- 固定環境: Windows CPU / Python 3.13.15 / NumPy 2.4.6 / torch 2.12.1+cpu、1 thread。
- tests/test_proposed_regression.pyのrun_case(sine2)を実行し、既存compareで33指標・離散列・coverageを照合した。exit 0、一致。
- config moduleの属性読込みを調査用に記録し、65種類を観測した。実行後にmodule classとtorch threadsを復元した。
- 設定の適用・復元やraw保存だけの読込みもあるため、実処理の参照元と静的ソースを併記した。
- 裸のglobal参照・引数既定値・定数・未通過分岐を完全に記録したという主張はしない。
- 旧package・golden・研究resultsの変更なし。要約と参照元はreference-inventory.md。

## Requirements Review Gate

- 結果: PASS（要件案の品質確認。人間の承認ではない）。主担当がkiro-spec-requirementsのEARSとreview gateを適用。
- 範囲: 6領域・22受け入れ条件。数値ID、条件・主語・応答、成功と失敗の観測可能性を確認した。
- 境界: SINE供給・実行順序を対象にし、学習・判断の内容、CLI・保存・描画・指標算出を対象外として明示した。
- エッジ条件: 不足・不正値・未対応方式、端数、区間長超過、処理部失敗、run間独立性を含む。
- 再現条件: 初期準備後の乱数状態から照合する。生成器単体のseedだけで最終goldenのstreamを再現できるとは扱わない。
- 命名: 新しいコード名は設計で具体化し、Lunaレビュー後に承認する。現段階のnaming.mdは既存名の利用範囲だけで、承認済み命名表ではない。
- 承認待ち: 要求。設計・task・新命名は未生成/未承認。過去の実装開始指示を本specの承認へ転用しない。

## 次の操作

requirements.mdを人間が承認した後、設計と役割・入出力・状態の命名表を具体化する。
命名だけの追加承認はLunaレビューと主担当の採否で行う。
設計・taskの通常の承認を経てから実装する。

## 2026-10-03: 要件承認と設計

- 要件: ユーザーの「承認します。次に進んでください。」により承認。spec.jsonに承認元と内容hashを記録した。
- 設計: light discovery、synthesis、design-review-gateを主担当が適用した。
- Gate: PASS。22件の要件IDを対応表に収録し、具体ファイル・入出力・状態所有・失敗動作・検証条件を確認した。
- 草案の修正: run乱数型をruntimeからexecutionへ移し、Protocolからruntimeへの逆依存を除いた。
- API probe: 固定環境でCPU torchの例外出口の乱数復元=true、RandomState.uniformとRandom.shuffleの利用可能性=true。新実装の検証結果ではない。
- 新しいproductionコード・テストは作成していない。命名承認と設計・task承認が必要なため。

### 命名レビューの実行状態

- 対象: naming.md revision 1。
- 依頼先: gpt-6-luna。spawn_agentの結果は`agent thread limit reached`。
- レビュー結果は得られていない。PASS・承認済み・指摘反映済みとは扱わない。
- 命名表と自己確認は用意済みだが、ユーザーが委任したLunaレビューの代替にはしない。
- 再開時は同revisionをLunaへ渡し、有用な指摘をdesign.mdとnaming.mdへ同期してから命名承認を記録する。
- 要件承認は維持する。設計は人間承認待ち、taskは未生成。実装可能状態はfalseのまま。
