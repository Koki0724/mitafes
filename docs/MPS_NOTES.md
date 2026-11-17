# MPS移植時の設計・検証メモ

初回のLinux上での検証記録。基本操作は [README](../README.md)、現在のモデル取得方法は [モデル重み](../models/README.md) を参照。

`realtime_counter` をコピーし、Apple Siliconの内蔵GPUをPyTorch MPSで使う実装にした。元の `counting_poc`、`realtime_counter`、`cpu_counter` は変更していない。このフォルダだけでモデル・プロファイル・評価用基準データを持つ。入力動画は別途指定する。元の作業フォルダ名は `mitafes_mac`。

**MacBook Airで動かす方針としてMPSは妥当。ただし、この作業環境はLinuxなので、Mac実機でのYOLO推論とリアルタイム速度は未検証。** MPSが利用できることと、必要な速度・精度を満たすことは別々に確認する。

## CUDAをMPSに変えるだけでよいか

PyTorchのモデル・入力をMPSへ移す方針でよい。Ultralyticsでは `model.predict(..., device="mps")` がその指定にあたる。この実装では重みも明示的に `.to("mps")` し、ウォームアップ後の推論器のデバイスも確認する。

MPSは**Apple GPU**を使う。映像の読み込み、追跡、通過判定、SQLite保存は引き続きCPUで行う。検出結果の `.cpu()` はNumPyベースの追跡処理へ渡すために必要なので残している。CPUだけでAI推論する選択肢や、MPSが使えない場合にCPUへ切り替える機能は設けていない。

このプロジェクトで合わせて変更した点は以下。

| 項目 | Mac用の対応 |
|---|---|
| CUDA用の環境 | Macでarm64の仮想環境を新規作成。CUDA用PyTorchや `onnxruntime-gpu` を使わない |
| 複数GPU・ONNX CUDA | 除去。人物・傘とも同じ `mps` で順に推論する |
| デバイスの確認 | Apple Silicon / arm64 Python / MPSの利用可否と実演算を起動時に確認。利用できなければ停止 |
| 数値精度 | 比較の条件をそろえるためFP32固定。FP16・量子化は今回の対象外 |
| 時間計測 | ウォームアップ後と各バッチ後に `torch.mps.synchronize()`。非同期のGPU演算が終わるまで待って計測 |
| メモリ計測 | macOSの `ru_maxrss` はバイト単位なので換算を修正。MPS割り当て量も別途記録 |
| パス | このフォルダのモデル・プロファイルを参照。Ubuntu固有のパスに依存しない |
| 入力・保存 | 有限のローカル録画で検証。フレーム順序・既定の解析頻度・チェックポイントを維持 |

人物・傘モデル、信頼度0.1、NMS IoU 0.7、追跡・境界の条件は既存版と同じ。ガラス側の画面下端の頭部も対象に含む。傘自体は人数に加算しない。

| カメラ | モデル | 入力長辺 | 解析頻度（25fps動画の場合） |
|---|---|---:|---:|
| ガラス側 | YOLO11m-pose + 傘用YOLO11s | 両方1280 | 2フレームに1回＝12.5観測/秒 |
| 見下ろし側 | YOLO11sで人物・傘を同時検出 | 1280 | 同上 |

25fps・stride 2なら、平均で**1観測あたり80ms以内**が追従の目安。ガラス側はその中に2モデルの推論、読み込み、追跡、保存が入る。2画角を1台で処理する場合はGPUとメモリを共有するので、単独測定の後に同時実行でも確認する。

## Macでのセットアップ

対象はApple Silicon、macOS 14以降、arm64版Python 3.11。この固定バージョン構成の配布パッケージにはPython 3.11用のmacOS arm64 wheelがあることを確認済み。Mac上でのインストール全体は未実行。

```sh
cd /path/to/mitafes_mac
python3.11 -c 'import platform; print(platform.machine())'
# arm64 と表示されること。x86_64ならRosetta経由のPythonを変更する。

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python doctor.py
```

Ubuntuの `.venv-counting*` はコピーしない。CUDA用の `--index-url` も指定しない。`doctor.py` はOS、Pythonのアーキテクチャ、依存バージョン、MPSの実演算結果を表示する。成功なら `mps_smoke_test: passed` で終了コード0。

`PYTORCH_ENABLE_MPS_FALLBACK=1` は自動設定しない。未対応演算が出た場合はまずそのエラーを確認する。手動でこの変数を設定すると一部演算がCPUへ回るため、その場合は純粋なMPS実行との比較条件が変わる。設定値は `config.json` と `summary.json` に残る。

## まず1画角・1分で確認

以下は、この作業フォルダと同じ配置でMacへデータを移した場合の例。動画は `mitafes_mac` に含めていない。別の場所へ置いた場合は `--glass-source` などのパスを変更する。

```sh
# 待ち時間を入れず、1分の映像を何秒で処理できるか測る。
python benchmark.py \
  --glass-source ../mitafes/data/1219/ch1_20261106_121900-124148_vscode.mp4 \
  --seconds 60 --compare-reference \
  --output runs/glass_speed

# 同じ区間を元映像の時刻どおりに供給して、処理遅延が増え続けないか測る。
python benchmark.py \
  --glass-source ../mitafes/data/1219/ch1_20261106_121900-124148_vscode.mp4 \
  --seconds 60 --pace --compare-reference \
  --output runs/glass_paced
```

見下ろし側だけを測る場合は `--glass-source` を `--overhead-source ../mitafes/data/1300/ch1_20261106_130000-133423_vscode.mp4` に置き換える。見下ろし側の既定開始位置は、既存の基準区間に合わせた **670秒**。別動画の先頭から測る場合は `--overhead-start 0` を付ける。

出力先は毎回新しい名前を使う。子プロセスのログは `<出力先>/glass.log` / `overhead.log`。測定中は各画角のログに約100観測ごとの進捗が出る。

## 2画角を1台のMacで同時に確認

```sh
python benchmark.py \
  --glass-source ../mitafes/data/1219/ch1_20261106_121900-124148_vscode.mp4 \
  --overhead-source ../mitafes/data/1300/ch1_20261106_130000-133423_vscode.mp4 \
  --seconds 60 --concurrent --pace --compare-reference \
  --output runs/two_cameras_paced
```

`--pace` を外すと同時実行時の処理余力を測れる。各プロセスは初期化・ウォームアップ後に入力時刻を開始するため、開始瞬間の厳密な同期試験ではない。両方が動く区間の性能を確認する。

1分で追従できたら `--seconds 600` などへ延長し、`--compare-reference` を外して継続測定する。MacBook Airで長時間運用できるかは、発熱による速度変化やメモリ使用量を含めて実機で判断する。

## 結果の読み方

`report.json` に各画角の結果がまとまる。

- `requested_duration_complete`：指定した長さの映像を処理できたか。短い入力動画で終了した結果を誤って成功としない。
- `summary.all_observations_processed`：予定した観測がすべて処理されたか。既定の1分なら750観測。
- `summary.realtime_factor`：映像の長さ÷処理秒数。**待ち時間なしの測定**で1以上なら、その区間は映像の時間内に処理できた。安定運用には余裕が必要。
- `summary.startup_seconds`：初期化・モデル読み込み・ウォームアップ等。処理秒数には含めず別記。
- `summary.latency_p95_seconds` / `latency_max_seconds`：`--pace` 時は入力予定時刻からカウント処理までの遅延。待ち時間なしの場合はデコード済みフレームの到着からの時間で、意味が異なる。
- `latency.windows`：映像の10秒区間ごとの遅延。等速試験で後半ほど増え続ける場合は処理が追いついていない。最初と最後のp95の差も記録するが、これだけで合否は決めない。
- `accuracy`：`--compare-reference` 指定時の通過・軌跡照合。人数の合計だけで合否を判断しない。

既存の基準区間は、ガラス側が **50左→右＋12右→左＝62通過**、見下ろし側が **3左→右＋13右→左＝16通過**。`accuracy.matched_trajectories` と未照合一覧で1対1の対応を確認する。MPSとCUDAは同じFP32でも演算結果が完全に一致するとは限らないため、実機でこの比較が必要。

`--compare-reference` は上記の既知の1分区間専用。他の録画・他の開始位置には付けない。この比較は既知データ上での回帰確認であり、未知の映像に対する精度評価ではない。

各画角のフォルダには `events.sqlite`、`events.csv`、`summary.json`、`config.json`、`timings.jsonl` を保存する。基準比較時は `tracks.jsonl` も保存する。RSSとMPSメモリは異なる指標なので、単純に加算して使用メモリとはしない。

## 通常実行・再開

```sh
python run.py --camera glass \
  --source /path/to/video.mp4 \
  --max-seconds 60 --pace --output runs/my_run

# 同じ動画・設定で続きから再開し、処理範囲を延長
python run.py --camera glass \
  --source /path/to/video.mp4 \
  --max-seconds 120 --pace --output runs/my_run --resume
```

`--start-seconds` は入力動画をスキップする秒数。出力のフレーム・時刻はその位置を0として数える。`--max-seconds` を省略すると末尾まで処理する。連続する分割録画は `--source part0.mp4 part1.mp4` の順に指定できる。同じ寸法・fpsで欠落・重複がないことが前提。

再開用データはこのプログラムが自分で生成したものを使う。再開時の `summary.observations` と速度は今回追加で処理した区間、通過数は保存済みを含む累計。ベンチマークは新規出力先で測る。

## この環境で確認した範囲

Linux上で以下を検証した。結果は [VALIDATION.md](../VALIDATION.md)。

```sh
python -m unittest test_streaming test_mac -v
```

保存済み検出を使う回帰テストで、既存の62＋16通過、傘監査、状態復元、分割ファイル境界の解析位置が一致することを確認。新しいテストではMPS不可時の停止、Rosettaの排除、MacのRSS単位、読み込み失敗の伝播、キュー停止、中断後のログ切り戻しと再開を確認する。これらのテストはMac GPUでの推論性能を検証するものではない。

実カメラのRTSP接続、通信断・再接続、画面表示や動画圧縮を含む速度、Mac実機のMPS推論・カウント精度、長時間安定性は未検証。本実装は録画を使ったリアルタイム相当の検証コードであり、RTSPの接続口は設けていない。

## 参照した公式情報

- [PyTorch MPS backend](https://docs.pytorch.org/docs/2.10/notes/mps.html)：MPSへのモデル・テンソルの配置。
- [PyTorch 2.9 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0)：MPSがmacOS 14以降を必要とする変更。古いMPS概要ページの12.3という記載だけで対象OSを決めない。
- [torch.mps.synchronize](https://docs.pytorch.org/docs/2.10/generated/torch.mps.synchronize.html)：GPU演算の完了待ち。
- [MPS environment variables](https://docs.pytorch.org/docs/2.10/mps_environment_variables.html)：CPUへの演算フォールバック等の設定。
- [Ultralytics予測API](https://docs.ultralytics.com/modes/predict/)：予測引数と結果形式。比較のため、このコードでは元と同じ8.3.235を固定している。
