# milvus_client.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from doc_processor import DocumentProcessor
from config import Config

class MilvusClient:
    def __init__(self):
        # 1. 连接Milvus数据库
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT
        )

        # 3. 创建/获取数据库“表”（Collection）
        self.collection = self._create_collection()
    
    def _create_collection(self):
        """创建Collection（如果不存在）"""
        if utility.has_collection(Config.COLLECTION_NAME):
            # 如果表已存在，直接返回
            connection = Collection(Config.COLLECTION_NAME)
            connection.drop()
            # return Collection(Config.COLLECTION_NAME)
        
        # 定义表结构：id（主键）、text（文本内容）、embedding（向量）
        print(Config.DIMENSION)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),  # 存储文本块
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.DIMENSION)  # 存储向量
        ]
        schema = CollectionSchema(fields=fields, description="知识库问答的向量存储表")
        # 创建表
        collection = Collection(name=Config.COLLECTION_NAME, schema=schema)
        # 创建索引（加速向量检索）
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",  # 计算向量相似度的方式（L2=欧氏距离）
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection
    
    
    def insert_documents(self, documents):
        """把拆分后的文本块插入Milvus"""
        # 1. 批量转换文本为向量
        texts = [doc.page_content for doc in documents]
        embeddings = DocumentProcessor.encoder(texts)
        
        # 2. 构造插入数据（text和embedding对应）
        data = [texts, embeddings]
        
        # 3. 插入数据并返回结果
        self.collection.load()  # 加载表到内存（检索前必须做）
        insert_result = self.collection.insert(data)
        self.collection.flush()  # 刷新数据到磁盘
        return insert_result
    
    def search_similar(self, query_text: str, top_k=3):
        """根据问题检索相似的文本块（top_k=返回前3个最相关的）"""
        # 1. 问题转向量
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        
        # 2. 配置检索参数
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # 3. 执行检索
        self.collection.load()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]  # 检索结果返回text字段
        )
        
        # 4. 整理检索结果（提取文本和相似度分数）
        similar_texts = []
        for result in results[0]:
            similar_texts.append({
                "text": result.entity.get("text"),
                "score": result.distance  # 分数越小，相似度越高
            })
        return similar_texts