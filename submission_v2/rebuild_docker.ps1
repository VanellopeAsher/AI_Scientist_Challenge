# PowerShell脚本：重新构建并启动Docker容器

Write-Host "停止现有容器..." -ForegroundColor Yellow
docker-compose down

Write-Host "重新构建Docker镜像..." -ForegroundColor Yellow
docker-compose build --no-cache

Write-Host "启动容器..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "查看容器日志..." -ForegroundColor Yellow
docker-compose logs -f

