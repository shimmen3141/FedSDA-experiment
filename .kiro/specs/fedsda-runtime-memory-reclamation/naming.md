# 公開名と役割

| 公開名 | 役割・既存名との違い |
| --- | --- |
| `runtime.process_memory_reclamation` | processの解放済みメモリ返却だけを扱う。モデルの寿命や学習メモリ管理、研究用のRSS診断を扱わない。 |
| `reclaim_unused_process_memory` | 対応環境で解放済み領域のOS返却を要求し、非対応なら何もしない。モデル・標本を解放する関数ではない。 |

既存公開名の変更はない。
