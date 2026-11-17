# モデル重み

重みはGitの管理対象外です。プロジェクト直下で取得・検証します。

```sh
python scripts/download_models.py
python scripts/download_models.py --check
```

取得元は [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)。固定したSHA-256を確認し、既存ファイルは上書きしません。

| ファイル | 用途 |
|---|---|
| `yolo11m-pose.pt` | ガラス側の人物・姿勢推定 |
| `yolo11n.pt` | ガラス側の傘、見下ろし側の人物・傘 |
| `yolo11s.pt` | 比較実験用。推奨プリセットでは使用しない |

チェックサムは [download_models.py](../scripts/download_models.py) に固定しています。オフライン環境では同じ重みをこのフォルダーへコピーし、`--check` で検証してください。
