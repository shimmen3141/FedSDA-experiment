# gpt-6-lunaによる設計・命名レビュー

2026-10-02〜03、ユーザー指定のgpt-6-lunaへ独立したコンテキストで読み取り専用レビューを依頼した。
対象は要求、設計、命名revision 3、研究記録、承認状態と必要な既存コード。
以下はレビュー指摘と、主担当が根拠を照合して決めた対応である。人間の設計承認を代替しない。

## 指摘と対応

| 優先度 | 指摘 | 確認した根拠 | 対応 |
|---|---|---|---|
| 高 | `reset_all_routing_state`が実態と違う | `clients/fedsda.py`の`_RestartingSoftRoutingFedSDAClientMixin._on_local_model_change`はSwitching・Meta-switchingをrestartしない | `restart_adahedge_preserve_switching`へ変更し、active set等のrestart範囲も説明 |
| 高 | 集約型の直接構築で組合せ検証を迂回できる | 個別型の値検証だけでは方式・機能間の条件を保証できない | 集約型の`__post_init__`から共通の組合せ検証を呼ぶ。検証側は集約型をimport・生成しない |
| 中 | 初回の値域宣言と後続カタログが未接続 | 型の宣言から公開項目定義への導出方法がない | 初回からフィールドmetadataへ値域・単位を宣言し、後続が同じ宣言を読む。説明取得APIの要求1.3は後続で完了 |
| 中 | 未収録条件がある初回型を`Resolved`と呼ぶと混同する | 候補optimizer、事前学習、データ生成等はまだ型に含まれない | 初回は`ValidatedExperimentRunSettingsSubset`。全条件が揃った後に完全な`ResolvedExperimentRunSettings`を追加し、部分型の実行・保存を拒否 |
| 中 | 小規模実行前の条件棚卸しの出口条件が弱い | 実行のための残り項目が「後続レビュー」とだけ記載されていた | 最初の対象を最終goldenのSINEケースとし、全参照条件・所有者・単位・新名を確認してから実行層を実装する条件を追加 |
| 高 | FIFO容量とFixed-Share時間尺度の受渡しが抜ける | `clients/fedsda.py`は`FIFO_BUFFER_SIZE`を両方に渡し、`expert_routing.py`はshareを`1/share_horizon`で計算する | `switching_share_horizon_samples`を予測設定へ追加。最終構成では容量から導出し、異なる明示値は拒否して旧結合を維持 |
| 低 | joint更新の説明が共有部だけの更新に見える | 既存joint学習は共有部・Adapter・headを更新する | 正式値の説明に概念固有部の共同学習を明記 |

コード参照の基底は`federated_drift_experiment/`である。

## 採用しなかった案

- 全routerをresetするようコードを変える案は採用しない。今回は旧挙動とgoldenの維持が目的である。
- 完全型の承認済み名を全面改名する案は採用しない。初回の部分型を明確に分け、完全型の意味は維持する。
- share時間尺度を新しい独立ハイパーパラメータにする案は採用しない。意味は別名で表すが、今回の移植では同値制約で旧実装の結合を保つ。

## 承認と検証

- 上記変更は設計・命名revision 4へ反映し、2026-10-03のユーザー返答で人間が承認した。
- 要求本文と命名revision 2の承認履歴は維持する。
- レビュー・反映は文書のみ。コード、golden、研究成果物を変更していない。
- 反映後の設計・命名について同じgpt-6-lunaが再確認し、主要指摘は解消したと判断した。
- 再確認時に本文と状態記録のrevision番号の不一致が指摘された。状態更新中の確認だったため、
  最終状態でspec.json・review.md・README・roadmapをrevision 4へ揃え、機械的に照合した。
- 軽微な追加指摘として、oracle概念別AdaHedge診断routerも保持対象であることを正式値の説明へ明記した。
