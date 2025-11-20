#!/bin/bash
# 重新构建并启动Docker容器

echo "停止现有容器..."
docker-compose down

echo "重新构建Docker镜像..."
docker-compose build --no-cache

echo "启动容器..."
docker-compose up -d

echo "查看容器日志..."
docker-compose logs -f

