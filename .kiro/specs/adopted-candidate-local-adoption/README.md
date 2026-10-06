# 採用候補のローカル採用

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

採否判定で採用と決まった候補について、クライアント内の状態更新を一つにまとめる。一時IDの採番、初期ローカル登録、候補学習量の計数、保留標本の学習標本への追加、現在の学習帰属IDの切替えを既存ownerで組み立て、変更記録を返す。旧FedSDAの前向き検証確定処理（_finalize_forward_validation）の採用分岐のうち、モデルとデータの状態更新が範囲。

採否の判定、判定記録（ProvisionalModelDecision）、切替位置・検出エピソード・適応イベントの記録、現在ID変更後の予測重み再始動などの通知、棄却/再利用/維持の分岐、候補sessionの開始・収集・破棄、計算量診断、通信、新client/全体runは後続である。
