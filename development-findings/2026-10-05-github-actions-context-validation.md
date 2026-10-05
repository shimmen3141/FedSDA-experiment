# Development finding: YAML読込みではGitHub Actionsのcontext制約を検出できない

- 観測日: 2026-10-05
- 観測した作業: Python品質CI導入
- 改善先: project
- 関連commit・artifact: `9eb10da`、[失敗したCI](https://github.com/shimmen3141/FedSDA-experiment/actions/runs/37313700464)

## 観測した事実

PyYAMLで設定を読み込み、トリガー・権限を確認し、独立レビューもPASSだった。
しかしpush後のCIはjobsが空のままfailureとなり、`gh run view`はworkflow file issueと表示した。
`jobs.<job_id>.env`で`runner.temp`を参照していた。
[公式context制約](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability)では
jobのenvにrunnerは許可されず、stepのenvなら許可される。

## 影響とworkaround

- 影響: ローカル検査が通ってもホストCIが開始しなかった。数値実装・goldenへの影響はない。
- その場のworkaround: `MPLCONFIGDIR`をpytestステップのenvへ移し、再pushで確認する。

## 仮説と改善案

- 仮説: 汎用YAML読込みと目視レビューだけではActions固有のexpression制約が漏れる。
- 改善案: context使用時は公式の許可範囲と照合し、CI導入完了前に実ホストでの実行結果を確認する。
  ワークフロー変更が増えた場合はActions専用validatorの導入を検討する。

## 改善結果

`.github/workflows/python-quality.yml`のenv参照を修正した。ホストでの再実行結果は
`docs/research/code-quality.md`の導入記録へ追記する。
