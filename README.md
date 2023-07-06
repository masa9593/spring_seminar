1. 宿題の目的
- ハイパースペクトル画像の理解を深める
- 特殊な画像に対する処理を学ぶことでpythonコーディングに慣れる
- ノイズ除去の手法について理解を深める

2. インストール手順
"https://docs.docker.jp/docker-for-windows/install.html"を参考にdocker desktopをインストールしてください。

3. 使用方法
- docker image build -t spring_image .
- docker run --name spring_container -v $(pwd):/app -it spring_image /bin/bash
- python dataset.py
- control + d で抜けられます（Macの場合）

4. 連絡先情報
slackのDMにお願いします。または"kimura.m.as@m.titech.ac.jp"までどうぞ。