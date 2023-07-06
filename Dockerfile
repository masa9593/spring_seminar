FROM python:3.8

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# ワーキングディレクトリの設定
WORKDIR /app

# 必要なファイルのコピー
COPY . /app

# ライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt
