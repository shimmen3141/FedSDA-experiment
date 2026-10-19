# Design

## Boundary
変更対象は既存HTMLと本specだけ。refactoringの実装・進行中文書、実験条件・golden・他の図示は変更しない。

## 実装との対応
参照は隣のrefactoring worktreeの現行実装（調査時HEAD 169bec6、既定化作業は進行中）。
- runtime/fedsda_final_configuration_run_settings.py: acceptanceはwithout_validation、candidate共有部はkeep_shared_features_fixed。
- runtime/alarm_change_interval_resolution.py: 再利用評価→初期値選択→train_candidate_on_alarm_change_interval→adopt_candidate_as_current_training_model。同じ応答内で完結しsessionを開始しない。
- learning/training/candidate_epoch_training.py・joint_model_parameter_update.py: 共有部をno_gradで推論しoptimizerを進めず、固有部だけ更新する。早期終了用の内部検証と、削除する将来標本による候補採否検証は別。
- runtime/held_model_training_request_handling.py: 通常共同学習はupdate_shared_features=True。
- learning/training/adopted_candidate_shared_feature_integration.py: 採用時のコピー関数は残るが、候補学習で共有部の値を変えない。図は値の不変を表現し、共有部自体を使わないとは描かない。

## 表示・遷移
既存の候補検証領域の外枠を新モデル作成へ転用し、共有部（固定）・固有部（学習）・即時追加を縦に並べる。固定共有部も順伝播に使うため、候補学習コマでは見せ、更新する固有部と区別した明示を置く。新規候補への入力はFIFO変化区間、追加先は更新用の保有モデル群、標本の確定先はモデル別ストア。

上段はフェーズ1〜4を保持し、5を「新モデル作成」（選択肢なし）、6サーバ共有を保持する。新規経路はC01（変化区間を渡す）→C02（固定共有部で特徴抽出・固有部を学習）→C03（即時採用・標本確定）。既存再利用はA07からサーバ共有へ。R01/R02は新規経路のみ。旧V/F frame・validation選択と専用線を除去し、採用状態はC03到達から決める。時点はfutureではなく現在の標本。

## 検証
変更前の受入検査は新規遷移・旧検証UI不在で失敗することを確認。変更後は全経路でactor/edge参照解決、順序、採用状態、フェーズ対象外、循環を検査する。1280×720・1440×900・390×844の表示と代表コマをブラウザ確認。モデル領域・サーバ関連frameの差分を照合する。実験コードのpytestは対象外。

## 追記設計（要求5〜7。前の配置・名称を置き換える）
- C01で更新用バックボーンと選択した1つの更新用モデル（模式例1′）から、候補用の独立したコピーへ矢印を示す。共有部は既存全モデルで同値、固有部は選択モデル1つからの初期値。モデル群全体を複製する意味ではない。
- 候補の固有部枠を3行の名称・残差アダプタ・分類ヘッドへ拡げ、「即時追加」の専用枠を除去する。
- C03は候補→新規モデルn′の「即時追加」線、更新用バックボーン→n′の共有接続、切出し標本→nの標本のデータ線を表示する。既存の帰属枠・1の標本を追加先として強調しない。
- 追加枠はモデルx1335/y245/w125/h60、標本x1335/y465/w125/h55。既存モデル2′とストア2の右、予測結合/監視領域の左の余白を使い、既存枠を移動しない。新規モデルはC03/R01だけ、標本枠はC03だけ表示し、全体図でも表示しない。共有接続もC03/R01だけ。
- コピー線は左右モデル間の余白やモデル間の間隔から下側へ折り返し、コピー先へ接続する。追加線は旧追加線の上側経路を使い、標本線は下側外周を使う。文字・矢尻と箱の交差を実ブラウザで確認して必要な座標だけ調整する。
- 標本の根拠：runtime/alarm_buffer_response.pyで変化区間のtraining_sampleをtupleにして渡し、alarm_change_interval_resolution.pyで特徴・ラベルを連結して学習、adopted_candidate_local_adoption.pyで同じ標本を新IDのtraining_sample_storeへappendする。図の標本枠はこの一時的な処理対象であり、別の永続ストアではない。
- 検査は現行版に対して旧名称/旧追加枠で失敗することを先に確認。全経路でC03/R01以外の追加枠のdisplay:none、追加先、コピー先、モデル共有接続、標本線の始点、文字の収まりを確認する。他frameの契約と操作は保持する。

## 追記設計：要求8〜9（共有矢印・同時表示の方針を置換）
- 新規経路はC00（標本受渡し）→C01（初期値コピー）→C02（候補学習）→C03（追加）。同じ警報応答の説明を分けるもので、実装上の待機や別標本を増やす意味ではない。
- C00はfifo-assign/assign-candidateを表示し、C01はcopy-backbone/copy-modelだけ表示する。既存の2本のコピー線は交差しない経路なので維持する。
- id=candidateの「変化区間の標本」枠だけをdataの青緑にし、assign-candidate/candidate-assign線と対応ラベルも同色にする。candidate-shell（新モデル作成の外枠）、candidate-backbone/head（モデルコピー）、stores/new-store（学習ストアの緑枠）の色は変えない。id=candidateはモデル候補本体ではなく、処理へ切り出した標本の枠を指す。assign-candidateの「変化区間」ラベルは枠自体の名称と重複し線にも重なるため除去する。
- new-model-shared線とラベルをSVG/frame双方から除去。C03はmodels/storesを追加で活性化、R01はmodelsを追加で活性化する。local-focusで配布時側は薄いままにする。新nモデル/ストアの限定表示は不変。
- 変更前検査は共有線の存在とC00不在で失敗することを確認。3画面・全経路で順序、コマごとの活性線、標本色とコピー色の違い、更新用モデル群/ストアの濃淡、限定表示と操作を確認する。

## 追記設計：要求10
サーバのregister_ready_client_modelsは正式IDを割り当てconfirm_held_model_registrationで保有モデルのID/帰属/記録を付け替える。この段階で配布時snapshotは更新しない。従ってR02もnew-modelを表示・活性化し、localOnlyへ追加する。変更は新規モデルのtitle font-size、表示限定CSSのR02追加、R02のactorsへnew-model追加、localOnlyへのR02追加の4箇所のみ。R02の文言や標本枠の表示条件は変えない。検査はR02の新モデルdisplayと配布時のopacity、文字のSVG bboxを確認し、他frame定義の不変を照合する。

## 追記設計：要求11〜13（中継枠を除去）
- 実装確認：released_pending_sample_assignmentは容量超過分をcurrent IDへ吸収してからFIFOから解放。alarm_buffer_responseはprepare_alarm_training_intervals（分割・前区間吸収）→resolve_alarm_change_interval（全保有モデルを区間評価→適合先へ標本吸収→学習帰属切替、または候補学習）を呼ぶ。新しい待機/データ保存場所は追加しない。
- assignment枠を除去し、旧領域の右側にpartition（中心1015,590）/reuse-decision（中心1255,590）の小さな菱形を置く。partitionはA03、reuse-decisionはA06/A07を中心に活性化し、他では薄い背景とする。右は警報、上はモデル群、下はFIFOで既存配置を動かさない。
- N04はreleased-sampleから0の標本へ直接。A04はFIFO前区間から0の標本へ、A07はFIFO変化区間から1の標本へ直接。0/1は既存の★と合わせた模式例。確定先はモデル別ストアであり、モデルパラメータへ標本を流す描写にしない。
- A03はFIFOと警報から区間分割記号へ。A05は変化区間→更新用モデル群の評価線、A06は評価結果→再利用判定へ。A07は★の切替と標本確定。C00はFIFO変化区間→候補の標本枠へ直接接続する。処理記号をすべてのデータ線の中継点にはしない。
- 旧assignment専用線とそのラベルを削除/置換する。frameのactor/edgeとFIFO区間表示の参照を追随させる。対象はN04/A03〜A07/C00。必要なら補足に「菱形は処理、枠は標本/モデルの実体」を短く追加する。
- 検査は変更前のassignment枠存在で失敗することを確認。3画面の全経路で新処理記号、データ線の始終点、残存参照なし、★、新n枠限定表示、循環/再生/補足を確認。対象外frameの契約不変を照合し、代表コマの文字・線の交差と接続位置を目視する。
