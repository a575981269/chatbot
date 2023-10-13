# 阶段1：构建依赖项
FROM python:3.8-slim AS builder

# 设置工作目录
WORKDIR /usr/src/app

# 复制requirements.txt并安装依赖项
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 阶段2：构建应用代码
FROM python:3.8-slim

# 设置工作目录
WORKDIR /usr/src/app

# 复制从第一个阶段复制的依赖项
COPY --from=builder /usr/src/app /usr/src/app

# 复制当前目录下的所有文件到工作目录（不包括依赖项）
COPY . .

# 暴露端口，假设应用运行在5000端口
EXPOSE 5000

# 定义容器启动后执行的命令
CMD ["python", "app.py"]
