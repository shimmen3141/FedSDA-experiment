# 調査記録

2026-10-06。既存cc-sdd方針とfable-methodに従い、現物を調査した。新libraryなし。
- 旧clients/base.py: registerはget_paramsの独立値とmodel_statsの生参照を保留。get_pending_model_infoは待機状態でも返し、消費しない。confirm登録の双方経路で消去する。
- 旧clients/fedsda.py: model_upload_delay_rounds>=1。新モデル作成時pending_ready=Falseと残回数を設定。promote_pending_to_readyは各ラウンド境界で一回減算し、0でBaseへ委譲。空/readyならno-op。最終構成ではこの派生の契約を用いる。
- 旧servers/base.py::_collect_pending_models: has_pending_modelでready確認→取得→送信/登録→登録確認。実送信・IDは後続。
- 新ModelAndClassLossStatisticsStoreは不変値を置換し、getで独立コピーを返す。保留に初期統計を複写すると古い統計になるためmodel_idを保持し上位で現在統計を解決する。正常同ID更新を旧生参照と照合。統計全置換/統合/ID変更時の保留対応は後続の登録/同期契約で扱い、今回一般化しない。
- 新snapshot generatorのplain dict[str,Tensor]を借用保持する。二重clone/forward/モデル全体保持は不要。frozenデータ記録の内部Tensor/辞書は借用で、深い不変型とは呼ばない。
- FedSDA固有の待機契約なのでmethods/fedsda/model_registrationへ配置し、learningから手法へ逆依存させない。
- 旧confirm_model_registrationは保留データを消去しても私有_pending_upload_roundsを残す。空ではpromoteがno-opで送信不可、次queueで新delayへ上書きされるため、この残値は送信時期に影響しない。新clearは0へ戻す。旧対照では保留データがない状態の有効残回数を0として比較し、私有の無効残値一致を要件にしない。
