# mitafes — Apple Silicon用 来場者カウンター

録画映像から人物を検出・追跡し、カウント線を通過した人数と方向を記録するPythonプロジェクトです。三田祭のガラス側・見下ろし側の2画角を対象に、YOLO11とPyTorch MPS（Apple GPU）を使用します。

このリポジトリは `mitafes_mac_base` のMPS版です。入力はローカルの録画ファイルで、RTSP・USBカメラ接続や画面表示は実装していません。検出・追跡・カウント・保存を行い、録画を等速で供給する検証にも対応します。

## できること

- **2種類の画角**：ガラス側はYOLO11m-poseで人物・姿勢を推定し、YOLO11sで傘を検出。見下ろし側はYOLO11sで人物・傘を検出します。
- **方向別の通過カウント**：追跡と画角別プロファイルを使い、左→右／右→左の通過を記録します。傘自体は人数に加えません。
- **処理速度・遅延の測定**：最大速度と等速入力、1画角と2画角同時実行を比較できます。
- **中断からの再開**：SQLiteとチェックポイントを使って同じ入力・設定で再開できます。
- **回帰確認**：保存済みの検出と基準軌跡を使って、追跡・カウントを検証できます。

## 動作環境

| 項目 | 条件 |
|---|---|
| ハードウェア | Apple Silicon Mac |
| OS | macOS 14以降 |
| Python | 3.11、arm64版（Rosetta経由のx86_64版は対象外） |
| 推論 | PyTorch MPS、FP32 |
| 依存パッケージ | [requirements.txt](requirements.txt) の固定バージョン |

MPSが使えない場合は起動時に停止します。推論をCPUへ自動で切り替える機能はありません。

## セットアップ

```sh
git clone https://github.com/Koki0724/mitafes.git
cd mitafes

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

# OS・アーキテクチャとMPSの実演算を確認
python doctor.py

# 公式のモデル重みを取得し、SHA-256を検証
python scripts/download_models.py
```

`doctor.py` が `mps_smoke_test: passed` と表示して終了コード0になれば、MPSの基本動作は確認できています。YOLO推論の速度・精度は別途実映像で確認します。

モデル重みと入力動画はリポジトリに含めません。モデルの配置とチェックサムは [models/README.md](models/README.md) を参照してください。手元にある重みは削除・上書きせず、そのまま利用できます。

## まず1本実行する

プロジェクト直下から実行します。動画は任意の場所に配置でき、出力先は未使用の名前を指定してください。

```sh
python run.py \
  --camera glass \
  --source /path/to/glass_video.mp4 \
  --max-seconds 60 \
  --output runs/glass_60s
```

見下ろし側は `--camera overhead` を指定します。プロファイルは元映像の画角・寸法に合わせてあるため、別の設置位置や画角で使用する場合は `reference/profiles/` の調整が必要です。

| オプション | 意味 |
|---|---|
| `--camera glass\|overhead` | 画角と使用するモデル・プロファイル |
| `--source PATH [PATH ...]` | ローカル動画。複数指定時は指定順に処理 |
| `--max-seconds 60` | 処理する映像の長さ。省略すると末尾まで |
| `--start-seconds 0` | 読み飛ばす秒数。出力上の時刻はそこから0で開始 |
| `--stride 2` | 解析するフレーム間隔。既定は2 |
| `--pace` | 映像の時刻どおりに供給して遅延を測定 |
| `--record-tracks` / `--record-detections` | 比較用の軌跡／検出データを保存 |
| `--resume` | 同じ出力先のチェックポイントから再開 |

25fpsの動画をstride 2で処理すると毎秒12.5観測です。strideを変更するとカウント結果も変わりえます。連続する分割録画には、同じ寸法・fpsで欠落・重複がないことが必要です。全オプションは `python run.py --help` で確認できます。

## ベンチマーク

```sh
# 最大速度：60秒の映像を何秒で処理できるか
python benchmark.py \
  --glass-source /path/to/glass_video.mp4 \
  --seconds 60 --output runs/glass_speed

# 等速入力：処理遅延が増え続けないか
python benchmark.py \
  --glass-source /path/to/glass_video.mp4 \
  --seconds 60 --pace --output runs/glass_paced

# 2画角を同時実行。ここでは両方とも動画の先頭を使用
python benchmark.py \
  --glass-source /path/to/glass_video.mp4 \
  --overhead-source /path/to/overhead_video.mp4 \
  --overhead-start 0 \
  --seconds 60 --concurrent --pace --output runs/two_cameras
```

`benchmark.py` の見下ろし側の既定開始位置は、元の基準区間に合わせた **670秒** です。別動画の先頭から処理する場合は、上の例のように `--overhead-start 0` を付けます。

`--compare-reference` は、元の指定された1分区間だけに使用します。基準はガラス側62通過（左→右50、右→左12）、見下ろし側16通過（左→右3、右→左13）です。合計に加えて軌跡の1対1照合も確認します。区間・入力条件は [MPS移植メモ](docs/MPS_NOTES.md) を参照してください。

## 出力と再開

各実行の出力先には次のファイルを保存します。`benchmark.py` はさらに `report.json` と画角ごとのログ・サブフォルダを作成します。

| ファイル | 内容 |
|---|---|
| `events.sqlite` / `events.csv` | 通過イベントと方向 |
| `summary.json` | 通過数、処理速度、遅延、メモリ等の集計 |
| `config.json` | 入力・設定・実行環境 |
| `timings.jsonl` | 各観測の時間計測 |
| `tracks.jsonl` / `detections.jsonl` | オプションで記録する比較用データ |

最大速度試験の `realtime_factor`（映像秒数÷処理秒数）が1以上なら、その区間を映像の時間内に処理できています。等速試験では待ち時間が入るので、`latency_p95_seconds` と時間帯ごとの遅延増加を確認します。モデル初期化は `startup_seconds` に別記されます。

```sh
# 先の60秒実行と同じ入力・設定で、120秒まで処理範囲を延長
python run.py \
  --camera glass --source /path/to/glass_video.mp4 \
  --max-seconds 120 --output runs/glass_60s --resume
```

再開では同じ入力・設定と、このプログラム自身が作成したチェックポイントを使用します。通過数は累計、観測数・速度は今回追加処理した区間の値です。速度比較は新規出力先で測定してください。

## テストと検証範囲

```sh
python -m unittest test_streaming test_mac -v
```

既存テストは、凍結した検出データでの62＋16通過、状態復元、分割動画、入力障害、中断後の再開、MPS不可時の停止などを検証します。モデル重みや入力動画は不要です。

初回の検証はLinux上で行われています。2026-11-17のリポジトリ整備時には、Apple Silicon Mac上で既存テスト13件、MPSの基本演算、依存関係の整合性、モデル2種の新規取得とSHA-256一致を確認しました。記録は [VALIDATION.md](VALIDATION.md) と [validation/](validation/) にあります。

GPUでの実映像推論・長時間の発熱・2画角同時運用の性能は、このMPS版について別途実機確認が必要です。録画の回帰テストだけで未知の映像への精度や終日運用を保証するものではありません。

## ファイル構成

```text
mitafes/
├── run.py                   # 動画入力・推論・保存・再開
├── engine.py                # ストリーミング追跡と通過判定
├── mac_runtime.py           # Apple Silicon / MPSの確認
├── doctor.py                # 環境診断
├── benchmark.py             # 1画角・2画角の速度／遅延測定
├── evaluate.py              # 基準イベント・軌跡との照合
├── requirements.txt         # 固定した直接依存
├── scripts/download_models.py
├── models/                  # モデル配置先（重みはGit管理外）
├── reference/               # 追跡実装、画角設定、回帰比較の基準
├── tests/fixtures/          # テスト用の保存済み検出
├── test_mac.py / test_streaming.py
├── validation/              # 初回Linux検証の記録
├── original_hashes.json     # 元プロジェクトの変更確認用記録
├── docs/MPS_NOTES.md         # 移植時の詳しい設計・操作メモ
└── runs/                    # 実行結果（Git管理外）
```

`.gitignore` で仮想環境、キャッシュ、ローカル設定、動画、モデル重み、実行結果を除外しています。`reference/` と `tests/fixtures/` のCSV・JSONは再現性のために管理します。
