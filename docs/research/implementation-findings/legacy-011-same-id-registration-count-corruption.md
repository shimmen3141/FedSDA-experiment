# LEGACY-011: 同一負IDの登録確認で件数が破損する

- 発見日: 2026-10-07。対象: 旧clients/base.py::confirm_model_registration、固定基準748c3aa。
- 発見spec: model-training-and-assignment-counts。種別: 不正登録通知での状態破損。状態: 再現済み・未修正。

## 再現と観測

current_model_id=-7、model_training_examples=defaultdict(int,{-7:3})、model_optimizer_steps=defaultdict(int,{-7:2})、model_concept_counts=defaultdict(Counter,{-7:Counter({1:4})})を持つ最小旧client（他ownerは空、pending_model_params=None）へ、実BaseClient.confirm_model_registration(client,-7)を呼ぶ。
2026-10-07基準venvで実行し、学習件数{-7:6}、step数{-7:4}、概念別辞書{}を観測。
整数の拡張代入は先を読み取った後に元をpopし再加算する。Counterは先を取得した後に同じ項目をpopし、辞書から消えたCounterへupdateする。

## 影響と今回の扱い

通常のサーバ通知は一時負IDから異なるグローバルIDへの変更であり、本再現は同一負ID通知という契約外入力。正常client経路・過去実験成果への影響は未確認で、過去成果が破損しているとは判断しない。
今回の新カウンタAPIは移管元と先の同一IDを状態変更前にValueErrorで拒否する。正常な異なるIDの加算/順序を維持する。旧production・goldenは修正しない。公開テストに実旧再現と新拒否の対照を残す。

## 将来の修正候補

旧登録確認の入口で通知先の非負ID・元との相違を検査する、または同IDをno-opと定義する。サーバ通知契約と全ownerの原子性を確認して別変更として決める。担当/修正commitは未定。正常登録・再通知・counterと概念診断の回帰が必要。
