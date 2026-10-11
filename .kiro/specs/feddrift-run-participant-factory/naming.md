# 公開名と役割

| 公開名 | 役割と似た名前との差 |
|---|---|
| runtime/feddrift_run_participant_factory.py | 1 runの全FedDrift参加者の初期準備。1 clientのassemblyとは別。 |
| FeddriftRunParticipantSettings | 初期準備のために既存client/server/事前学習設定を束ねる。 |
| FeddriftRunParticipants | 準備済み全client/serverへの参照。FedSDAの操作Protocolを持つRunParticipantsとは異なる。 |
| FeddriftRunParticipantFactory | 事前学習から全参加者準備までの入口。 |
| FeddriftRunParticipantFactory.validate_configuration | 参加者を作らず設定だけ検査する。 |
| FeddriftRunParticipantFactory.prepare_run | 新しい参加者集合を生成する。 |
| FeddriftRunParticipantFactory.prepared_run_participants | 直前に成功した集合への参照。 |
