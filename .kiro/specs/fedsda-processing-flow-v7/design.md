# Design Document

## Overview / Boundary Commitments
v6から別のstandalone HTMLへ派生する。実装の意味はrefactoring側HEAD c4222ffを読取り確認した。公開Python APIは変更しない。元HTML/spec、実験コード/共有steeringには書き込まない。

## 短縮と行間（要求1）
新コマF01（epoch=future,stage=5）をP01〜P07の代わりに1つ置く。input/outputを濃くし、タイトル「次の標本も通常どおり予測」、説明で同じ混合予測の後に正解を用いて検証へ進むと示す。V01以降は保持する。候補分岐3組だけ6コマずつ減り、18組合せは490→472コマ（対象外値重複込み）。
caption small margin-top=3px、strong line-height=1.35、small=1.4。caption行はdesktop42px、狭幅58px、phone72pxを上限目安に実測で収まりを確認する。図下余白20pxは維持。

## モデル状態の過渡表示（要求2）
v6は親actorのopacity .28sと子stateのopacity .25sが逆方向へ同時遷移し、local-focus解除で子sync .22→1が親1→.28より先に増えて実効濃度が上がる。子stateはtransition:noneとする。非活性へ移る時のfocusは直前のfocusを維持し、モデル群が非活性の間はそのまま背景に置く。次にモデル群が活性になる時だけ、そのコマのlocal/sync/bothへ即時切替える。親のfadeと矢印の流れは残す。全体図はbothへ戻す。これにより非活性中にsyncを明るく戻さない。

## クロス評価（要求2）
独立actor cross-evaluationをモデル欄のバックボーンと各モデルの隙間（x450,y211,w855,h28）へ置く。通常/全体はopacity0、R05/R06だけ1、transitionなし。文言「評価用コピー（サーバの集約後モデル）」とし、通常モデル状態へ上書きする図にしない。
R05はclusteringとcross-evaluationだけを濃くする。R06はcross-evaluation/eval-data/clustering、モデル群の更新用側も正誤比較の診断用として濃くする（local-focus）。予測用の配布時側は薄い。request/result線は新actorの左端へ接続（requestは450,225、resultは450,235から）。旧点420,270を新copy宛てへ変え、他ノードの内部を横断させない。captionと補足で主対象と診断を区別する。
根拠: model_cross_evaluation.pyがglobal_model_repositoryの集約後snapshotを渡す。client_model_cross_evaluation.pyが独立classifierへloadして評価し、異なるID対ではheld_target_model_training_state.classifierの正誤も比較する。synchronized prediction snapshotは読まない。

## 候補と参照（要求2）
C01にfifo actorとfifo-assign線を加え、内部の変化区間側セルだけ濃くする。既存assign-validation線からcandidateへ接続し、captionをFIFO変化区間で新規候補を初期学習とする。C02は更新用側→固定参照のcopyとし、local-focus/backbone/param-localを含める。「固定参照」は既存モデル群の警報時点の独立写しで、配布時写しとは別。candidate表示を「新規候補」「学習後は固定」へ。説明は短くし詳細は補足へ。
根拠: alarm_buffer_response.py→prepare_alarm_training_intervalsの変化区間をresolve_alarm_change_intervalへ渡し、post_alarm_candidate_validation_session_start.pyがtrain_candidate_on_alarm_change_interval後にfix_reference_models_at_alarmを作る。

## 検証（要求3）
v6 CDP基盤をv7へ派生。各画面1280×720/390×844/1440×900で18組合せ・472コマを期待列照合。全コマで固定bbox・見出し濃淡・FIFO/通信/コピー・caption文字の収まりを検査。C01は変化区間セル/R05は独立copy/R06は診断local、F01は短縮接続を検査する。既存モーダル/循環/共有例も保持検査。
別の通常モーション検査はtransitionを無効にせず、steady N01→N02（8→9相当）、R12→R03（13→14相当）、N05→R12→R03、R10→R11、逆送りを、2つのRAFを挟み安定させた後に複数時点（0/16/32/64/120/200/320ms）の親×子実効opacityで確認する。配布時非対象のまま退場する遷移で、元の値を上回らないことを検査。モデル群の次回活性と全体へ戻る時は期待focusへ戻る。通常アニメーションと低減モーションで代表PNGを確認する。
実装前の独立CLI仕様レビュー、全task後に別sessionの実装レビューを受ける。
