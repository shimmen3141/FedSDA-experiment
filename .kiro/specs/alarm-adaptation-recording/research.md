# 調査

## 境界と実旧oracle

旧clients/fedsda.py::_resolve_driftの5経路のイベント・切替位置・再利用計数を、完了済みalarm-response-completionのtest helperで取得する。既存helperはイベント記録/reset/clearを差し替えず実旧を走らせる。新しい旧メソッドを導入しない。

## 予測通知と保存診断（2026-10-08、読取り調査）

- clients/fedsda.py 2052–2062の通知はglobal/context/shadow-meta AdaHedgeと任意active setを再始動する。switching_expert_routerは対象でない。
- 最終予測は同1763–1765のSwitching重み/scoresを使う。expert_routing.py 164–167の再始動はAdaHedgeのconcept_restart_countを増やし、experiment.py 1554–1556がNPZのrouting_concept_restart_countsへ保存する。
- AdaHedgeのproposalはfedsda.py 1609–1614、global_scoresは1651–1653、診断は1830–1847。experiment.py 1189–1194のrouting_switching_global_gain_rateに影響する。LOO診断もfedsda.py 1794–1797でproposalをfallbackに渡し、diagnostics/routing.py 328–342で残存Switching重みがほぼゼロのとき使用する。
- test_proposed_regression.py 34–42とproposed_regression_golden.jsonの設定はepisode無効、activation always、active set all。無効episodeのmark_operationはdetection_episode.py 38–40で状態を更新しない。
- 最終3goldenのMETRICS/TRACESは上記の再始動回数/global gain/LOOを比較しない。golden成功だけでは通知欠落を検出できない。

結論: 最終予測だけのためのFixed-Share再始動は不要だが、保存診断を維持する通知は後続の診断ownerへの接続として残す。今回の記録ownerは予測方式へ依存しない。episode無効の最終構成の組立と、任意episode方式の移植範囲は後続で区別する。アルゴリズム変更や診断の削除を今回へ混ぜない。

旧のイベント記録/reset順との差はdesign.md 4節。新recordの作成はrecord ownerの更新より前とし、入力失敗の部分記録を残さない。旧アルゴリズム改善は別候補として扱う。
