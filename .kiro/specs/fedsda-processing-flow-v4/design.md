# 設計

## スコープ
v3を入力とする表示の派生。旧版とv3の完成記録は書換えない。SVG/inline JS/CSSだけをv4へ生成する。

## 実装確認（c4222ff、refactoring worktreeは読取りのみ）
- runtime/fedsda_run_client.py: apply_global_model_distribution（396行付近）で手元の保有状態へ反映し、直後の413〜417行でcapture_held_modelsを呼ぶ。
- methods/fedsda/prediction_combination/synchronized_prediction_model_snapshot.py: capture_held_modelsで全held classifiersを1回のdeepcopyで写す。共有部の共有関係を保つ。
- runtime/server_model_registration_and_aggregation.py: register_ready_client_modelsはID採番のみ（パラメータ送信なし）。aggregate_client_models_into_global_modelsで学習件数>0の非負IDモデルを参加させ、共有部をclientごと、固有部をモデルごとに送信/集約する。
- runtime/server_round_synchronization.py: 登録→集約→評価/統合→配布→再較正。R12とR03は同じ関数内の送信と集約を図で別コマに分解する。

## コマ列と状態
buildの共有部分にR12,R03,...R11を置く。採用した場合だけその前にR01,R02。R12 actors=models,backbone,param-local,aggregation、edge=upload、local-focusを適用。R02は登録のみでuploadを消し、R03は集約のみ（旧register-aggregate接続は不要）。
R09は既存の青い配布矢印を更新後領域へ。R10はbackbone/models/param-syncを活性化しsync-focusで保存先を濃くし、右の共有部から左の共有部へのcopy-snapshot接続（1102.5,205→1102.5,225→652.5,225→652.5,205）と「全モデル＋共有部をコピー」のラベルを表示する。共有部1個だけのコピーではなくモデル状態全体の写しであることをcaptionにも書く。
current-ringとstarをstate-localのwrapperへ入れ、R10だけvisibility:hidden。他は現行の子だけopacity1、wrapperに更新後と同じ減光。adopted badgeの星もR10だけ消す。値や学習帰属は変えない。

## レイアウトと表示
- phase 1紫/2teal/3red/4green/5amber/6blueの6色。stageとdotはdata-stageにより同じ--tone/--tint/--tone-borderを設定し、caption番号の色も同じ。epochは時間の区別として残す。
- stage-startはi===0またはstageの切替時。枠は3px。最初だけmargin-left:0。
- prob-sync/prob-localの下段text/tspanを3個の独立textに置換し、箱を左右5ずつ広げ幅115にし、x=左端+22/57.5/93、y=207。タイトルと短い縦接続の中心は維持し、配布時入口の矢印終点だけ左5へ追随する。各textは同じmodel-id/state/symbol属性を持ち、データ対応検査を再利用。文字bboxの左右がbox+6..box幅−6内にあると検査する。
- 候補検証shellは40,485,300,260、タイトル56,475、candidate/references/loss-log/decisionはtranslate(20,10)。data-boxも実座標へ追随。assign-validation/copy-reference/adopt/validation-return/future-validationと内部3線を追随させる。future-validationは末尾を(190,763)→(190,745)とし、軸長15以上を保つ。
- uploadは更新後bank下側からサーバaggregationへ（1102.5,395→1102.5,415→335,415→335,230→290,230）。R02と同じ矢印を使わない。下りと区別できるlabelを近くに付ける。

## 受け入れ証拠
既存headless Chrome auditをv4専用に派生し、全2310コマで配色一致、初回stage-start、各結合記号のbbox、候補枠の余白、R12の全共有枝存在とuploadの向き、R09→R10の順序、R10copyの右→左、マークvisibility/実alphaをassert。既存の全コマ配置/操作/箱横断/確率入力を再検査。代表PNGはprediction、steady-upload、steady-distribute、steady-snapshot、candidate、overview。source行の根拠を独立CLIに渡す。Pythonやgoldenは変更/再実行しない。

## R12/R03の抽象度と組合せ
- R12とR03はaggregate_client_models_into_global_modelsという同じ関数をネットワークの役割から図解する2コマ。R12は参加パラメータ/重み付け件数が集約の入力になること、R03はそれらの重み付き平均を表す。実コードは同一processで計算し、検査/計算後に通信量を記録する。R12用の新しい処理や、Python命令が文字通り送信→計算となる順序を主張しない。R12のnoteにも「同じ集約処理の入力を図示」と記す。対象は非負IDかつ学習件数>0。
- 108組合せ=3予測×3警報分岐×2再利用/候補×3検証結果×2共有選択。対象外の選択値の重複も検査に含む。うち共有表示は54組合せで、R12を1回ずつ追加した総2310コマ。

狭幅fitではfontのピクセル丸めで数学用文字のbboxが通常幅より広く計測されるため枠内を検査し、明示拡大後に6図単位の余白を追加検査する。通常画面では全コマで6を維持する。
