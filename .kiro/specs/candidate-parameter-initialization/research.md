# 調査: 候補パラメータ初期化

## Summary
旧clients/fedsda.py:574–605の初期化用snapshot選択を独立移植する。最終goldenの方式はbest_candidate。
今回は調査で新しい旧不具合を観測していない。既存LEGACY-001〜008の状態は変更しない。

## Research Log
旧currentは現在の学習先、best_candidateは評価済み列のlossだけでminを取り同率先着、空ならcurrent。averageは全models.values()順のstack→meanで、整数・真偽値は先頭clone、空はNone。
旧models.py:122–126 get_paramsはstate_dictのdeepcopy。Shared/Residual modelも同APIを継承し、初期化でbackboneを分割しない。
旧_resolve_drift:795–813の評価列はhistorical baselineが0のモデルを含まないが、再利用不適合なモデルは含む。新部品は評価列の生成や適合判定を繰り返さない。
旧呼出後の生成/set_params/reset_optimizerとSharedの採用時attachは別責務。
既存新srcにparameter snapshot型はないためplain dictで明示境界を作る。新Settingsは初期化元専用とし、採否設定を意味変更しない。
読取専用調査agentはモデル生成なしstubで旧三方式/空/同率/整数bufferコピーを実測した。新数値の照合は実装testで改めて行う。

## Design Decisions / Synthesis
一般化: 選択と平均を「完成した独立snapshot作成」という一出力契約にまとめる。独立plan APIを増やさない。
採用: 既存設定field validatorと旧torch stack/meanをそのまま使う。新ライブラリを追加しない。
簡素化: model factory、callback、汎用state型、共有部専用初期化を作らない。
専用設定には省略defaultを作らず上位が明示する。将来runの標準choiceと型のdefaultを混同しない。

## Risks and Verification
実モデル通常stateはCPU float32。整数/真偽/複素buffer分岐は旧helperの契約照合として扱い、最終モデルの通常stateと混同しない。
全入力を先行検査して非有限やschema違反を拒否する。これを旧production修正や通常実験不具合の証拠とは扱わない。
モデル順、parameter順、浮動演算順、独立storage、grad/RNG/default環境を確認する。goldenは変更しない。

## 参照した手順
fable-methodの調査→観測可能な単位→実行検証、cc-sdd requirements/design/tasks/impl/review/verify/validateとnaming-reviewを使用。
日本語を含む編集はapply_patch、ASCII metadataだけPythonで扱う。
