# optical_flow
入力動画に対して疎なオプティカルフロー（もしくは密なオプティカルフロー）の解析を行う


## install

```
pip install requirements.txt
```

## execute
```
python --movie_path {解析用動画} --model {使用モデル} 
```

```
・その他コマンド
--show : 画面表示するかしないか（デフォルト:False）
--output_dir_path : 結果画像出力を行うパス（設定されていない場合は出力を行わない）
```