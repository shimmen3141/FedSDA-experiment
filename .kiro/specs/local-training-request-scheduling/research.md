# 調査根拠

## 軽量discovery
旧固定基準748c3aaのclients/base.py:198–212が要求一件追加→間隔判定→flush→成功後clearを持つ。
clients/shared_backbone.py:234–239はUPDATES_PER_SAMPLE×count_multiplierをrange回数とする。
clients/fedsda.py:485の警報前flush、525のtrain_stepは呼出し位置の根拠だが、今回移植しない。
clients/feddrift.py:114のk_stepsによる直接学習も別経路とし、本specに逐次要求を強制しない。

旧_pending_updatesはコメントどおり学習要求の件数であり、実optimizer update件数ではない。
今回の新名は要求件数と共同更新試行回数を区別する。
参加者不在や予算0でも旧train_all_held_modelsが正常returnすればclearされ、例外ならclearに届かない。
完成済みheld-model-joint-training-iterationsも空試行をskipし、失敗後の既済更新を巻き戻さない。

## 境界候補
独立した固定設定と単一所有の保留counterだけを追加する。計画返却により外側が実学習し、成功確認で消化する。
任意callback/registry・NN依存・時間/診断・状態rollback・並行ticketは導入しない。
設定はoptimizer設定型と同様の機能別単独型。初回ValidatedExperimentRunSettingsSubsetは完全run型ではないため変更しない。
上位の完全設定/CLI/プリセット組立は後続。新設定のmetadataを宣言し、共通値検査を再利用する。

## 使用手順と根拠
fable-method、kiro-spec-init/requirements、既存policy・steering・命名reviewを参照。
新外部依存はない。旧sourceと新既存設定/検査・共同学習の公開契約を調査し、独自の設定frameworkは不要と判断。
主担当が全条件の入力/出力/所有/異常時の維持を要求gateで確認した。
今回の通常経路調査では新たな旧不具合はまだ観測していない。発見時はimplementation-findingsへ記録する。
