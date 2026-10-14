# Research & Design Decisions

## Summary
- Feature: fedsda-default-waited-registration-and-alarm-wait。
- 2026-10-14、ユーザーが、FedSDAの既定を案Y（待つ登録＋002を直した側＋待つ警報）にすると決めた。

## Research Log
- 根拠の数値: [LEGACY-016の記録](../../../docs/research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)の「本規模の比較（待つ登録と待つ警報を含む8通り）と、案Yの判断」。
- ADWINと待つ扱いの組合せは、`LossChangeDetectionSettings.__post_init__`が拒否する（ADWINは、検知したときに窓の古い側を削るので、待っても合図が続かない）。ADWINの2構成は、検出の設定を`replace`で替えて作るので、既定が待つ扱いになると、そのままでは組立てが失敗する。
- 待つ扱いでは、保留の容量が最小件数以上でなければならない（`FedsdaRunClientSettings`の検査）。最終構成の値で満たされること（本規模の比較で、待つ扱いの全runが実行できた）。
- 基準版4の保存した設定は、警報の扱いの項目を`raise_alarm_immediately`として持つ（版4は、待つ扱いの実装の後に作った）。

## Design Decisions
- ADWINの2構成は、警報を即時に明示し、登録の扱いは既定（待つ登録）を引き継ぐ。
- 版の差の事象は、待つ扱いの分岐の条件そのもの（警報が上がり、保留の件数と推定変化区間の小さい方が最小件数より短い）の回数にする。応答の種類（短い警報）で数えると、候補検証を保持している間の場面（応答の種類が別になる）を数え漏れる（仕様レビュー1回目の指摘で改めた）。

## Risks & Mitigations
- 版5で、版4の必須の経路が通らなくなる → 版4のときと同じく、版5だけにseedの違うケースを足す。
- 既定に依るtestの取りこぼし → 前の2回と同じく、全pytestで失敗したものから拾う。

## References
- [pending-model-registration-after-waiting-as-current-model](../pending-model-registration-after-waiting-as-current-model/design.md)、[alarm-wait-for-minimum-change-interval](../alarm-wait-for-minimum-change-interval/design.md)
- 共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
