# 本地知识库的搭建--使用的是milvus向量数据库
##    下载
milvus数据库需要结合docker（下载docker desktop）使用，可下载官方的配置文件  
   curl -L https://github.com/milvus-io/milvus/releases/download/v2.4.3/milvus-standalone-docker-compose.yml -o docker-compose.yml
在docker运行状态下，切换到docker——compose.yml所在的文件下，运行  
  docker-compose up -d  
查看milvus数据库是否正常运行  
  docker ps命令下：milvus-standalone、etcd、minio 三个容器的 STATUS 为 Up，表示 Milvus 部署成功  
##    停止和卸载
docker-compose stop --停止milvus服务
docker-compose down --电脑上卸载milvus相关服务
