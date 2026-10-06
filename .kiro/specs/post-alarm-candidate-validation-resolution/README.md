# 警報後の候補検証の確定

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

警報後の損失による候補採否の評価結果を受け取り、結果に応じたクライアント内の状態更新を一つ選んで適用する。候補の採用（新モデルとして登録し学習帰属を切替え）、保有モデルの再利用（学習帰属を切替えて保留標本を吸収）、現行モデルの維持、候補の棄却（どちらも現行モデルへ保留標本を吸収）の4通り。旧FedSDAの前向き検証確定処理（_finalize_forward_validation）のうち、最終構成の方針（forward_persistent）が通る分岐の状態更新が範囲。適用した結果の種類・保留標本の帰属先・学習帰属の変更記録を返す。

採否の評価そのもの（post-alarm-candidate-loss-evaluationで完了）、損失の収集、判定recordの保存、切替位置・検出エピソード・適応イベントの記録、学習帰属変更の通知（予測重みの再始動）、候補sessionの開始と破棄、参照モデルも学習させる方針（旧shadow_tournament）の値反映、計算量診断、通信、新client/全体runは後続または既存specである。
