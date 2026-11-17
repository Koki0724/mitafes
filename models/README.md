# モデル重み

このフォルダに以下のUltralytics公式重みを配置する。重み自体はGitの管理対象外。

プロジェクト直下で実行する：

```sh
python scripts/download_models.py
python scripts/download_models.py --check
```

取得元は [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)。
Python標準ライブラリだけでダウンロードでき、SHA-256が一致した場合だけ保存する。
既存ファイルは検証のみ行い、不一致でも上書きしない。

| ファイル | 用途 | サイズ |
|---|---|---:|
| `yolo11m-pose.pt` | ガラス側の人物・姿勢推定 | 約42.5 MB |
| `yolo11s.pt` | ガラス側の傘、見下ろし側の人物・傘 | 約19.3 MB |

元の検証に使用した重みのSHA-256：

```text
29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2  yolo11m-pose.pt
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5  yolo11s.pt
```

ネット接続のない環境では、同じ重みをこのフォルダへコピーして `--check` で確認する。
モデルの利用条件は配布元のライセンスを参照する。
