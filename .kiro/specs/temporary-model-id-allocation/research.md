# 根拠と判断

- 旧BaseClient.__init__: next_temp_id = -100 - client_id。旧_alloc_temp_id: 現在のnext_temp_idを返し、1減らす。呼出しは3箇所（base._spawn_new_model、fedsdaの前向き検証採用、holdout検証採用）で、いずれも登録の直前。棄却時は採番しないので、IDは採用した順に連続する。
- 採番値はクライアント内のID。client kの最初のID(-100-k)は、client 0のk+1個目のIDと同じ値になる。旧はクライアントごとの辞書で一時IDを扱い、サーバは回収時に正式IDを採番して通知する。旧サーバの回収経路（servers/base.pyのpending回収と登録確認の通知、servers/fedsda.pyのモデル登録）は、clientごとのpending情報を順に回収し、サーバ側で正式IDを採番してそのclientへ確認を返す。確認した範囲（federated_drift_experiment/servers/とclients/base.py・clients/fedsda.pyの採番/登録/確認）では、サーバが一時IDをkeyに複数clientのモデルを扱う経路はなく、この回収経路で同値一時IDの衝突は起きない。診断・成果物保存（experiment.py等）が一時IDをclient横断で集計するかは調査範囲外。本specは旧の採番列を維持し、値の衝突回避を追加しない。
- 旧client_idは実験runnerがrange(クライアント数)から渡す非負int。新は既存data層と同じくexact int非負を要求する。
- 採番は登録・現在ID・統計のいずれのownerにも属さない独立した単調counterなので、単独のownerにする。採番済みIDの保有確認や再利用は行わない（登録側が使用済みIDを拒否する）。
- Python intは任意精度で、採番回数の上限を設けない。
- 対照oracle: 実旧`BaseClient(client_id, {0: object()}, verbose=False)`で実__init__が完了し、実_alloc_temp_idを呼べることを確認した（client_id=3で-103,-104,-105、次-106）。モデル・設定の準備は不要。testは初期値と採番の式を複製せず、この実旧呼出しの戻り値だけを正解にする。
- 全回帰をfeatureのtaskに含める理由: worktreeのAGENTS.mdは変更後の回帰実行と既存手法の値の不変確認を求め、これまでの全specが最終taskで固定環境の全pytest（旧11・最終3golden）を記録している。新moduleと依存guardの追加が既存testへ影響しないことを同じcommitで示す。判定基準はsteering/agent-handoff.md。
- 登録への接続test: 採番ownerの責務ではないが、採番値が登録APIの一時ID契約（exact int・負・未使用）を連続して満たすことを、後続の採用接続specの前に確認する。testからruntimeを呼ぶだけで、採番moduleはruntimeへ依存しない。
