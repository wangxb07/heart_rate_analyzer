# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖和 Python 开发工具
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 和安装基础包
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建上传文件夹
RUN mkdir -p uploads

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "app.py"]
