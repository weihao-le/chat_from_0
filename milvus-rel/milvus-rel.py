from pymilvus import connections,Collection,FieldSchema,CollectionSchema,utility,DataType
import numpy as np


###数据库的连接
connection = connections.connect(
    alias='default',###连接名称
    host='localhost',
    port=19530
)

##查询数据--集合
utility.has_collection('collecction_name')


##创建集合（字段框架->集合框架->创建集合）
#字段框架
fields = [
    FieldSchema(name='id',dtype=DataType.INT64,is_primary=True,auto_id =True),
    FieldSchema(name='embedding',dtype=DataType.FLOAT_VECTOR,dim=512),
    FieldSchema(name='text',dtype=DataType.VARCHAR,max_length=200)
]
#集合框架
schema = CollectionSchema(fields,description='知识库问答的向量储存表')
#创建集合
collection = Collection(name='collection_name',schema=schema)


##集合的修改
data = [
    [np.random.rand(512).tolist(),np.random.rand(512).tolist()],
    ['第一条数据','第二条数据']
]

#插入数据-实例化集合后能插入
'''
#1.数据要和集合创建的数据顺序保持一致
collecction = Collection('collection_name')
collection.insert(data)
#2.数据直接和字段对应传入数据
collection.insert(
    data={
        'text':['第一条数据','第二条数据'],
        'embedding':[np.random.rand(512).tolist(),np.random.rand(512).tolist()]
    }
)
#3.刷新数据
collecction.flush()
'''


#删除数据--通过某一字段（可以使id，id可以是自己制作也可以是自动生成，自动生成的id在插入的时候可以查询，也可以通过字段进行删除，例如text字段）
'''
#1通过id进行删除，通过id构建删除条件->删除->刷新
colection = Collection('collection_name')
ids_to_drop = [101,102]
delete_expr = f'id in {ids_to_drop}'
del_result = collection.delete(delete_expr)
collection.flush()

#2通过某一字段进行删除,字段支持支持 in、==、!=、>、<、like 等运算符，以及 and、or 逻辑组合
collection = Collection('collection_name')
delete_expr = 'text like "%过时的数据%"'
del_result = collection.delete(delete_expr)
collection.flush()
'''


#增加字段，不能插入主键字段和向量字段，且插入字段后的数据直接的数据为None:构建字段框架-->插入字段
'''
field = FieldSchema(name='name',dtype=DataType.INT64)
colection = Collection('collection_name')
colection.add_filed(field)
#删除字段--milvus不支持删除字段，只能创建新的集合，将原始数据迁移过去
new_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    ],
    description="不含 timestamp 字段的新集合"
)

# 3. 创建新集合
new_collection_name = "new_text_embeddings"
if not utility.has_collection(new_collection_name):
    new_collection = Collection(name=new_collection_name, schema=new_schema)
else:
    new_collection = Collection(new_collection_name)

# 4. 从旧集合迁移数据（只查询需要保留的字段）
old_collection = Collection(Config.OLD_COLLECTION_NAME)
old_collection.load()  # 加载旧集合到内存

# 查询旧集合数据（只取需要的字段）
results = old_collection.query(
    expr="",  # 空表达式表示查询所有数据
    output_fields=["embedding", "text"]  # 只获取需要保留的字段
)

# 5. 将数据插入新集合
if results:
    data = {
        "embedding": [item["embedding"] for item in results],
        "text": [item["text"] for item in results]
    }
    new_collection.insert(data)
    new_collection.flush()
'''


#查询数据
#1标量查询（和之前删除数据相同）
#2向量查询（通过向量相似度进行查询）将数据加载-->配置检索参数（需要创建索引）-->向量检索
collection = Collection('collection_name')
collection.load()
query_embedding = np.random.rand(512).tolist()
search_params = {
        'metric_tyep':'L2',
        'params':{'nprobe':10}
}
results = collection.search(
    data = [query_embedding],##查询向量，支持批量
    anns_field = 'embedding',#查询向量字段
    params = search_params,##字段检索参数
    limit = 5,#返回相似的条数
    output_fields = ['text'],##输出的字段
    expr = ''##可见的一些过滤条件
)

# results 是列表，每个元素对应一个查询向量的结果
for hits in results:
    print(f"与查询向量最相似的 {len(hits)} 条数据：")
    for hit in hits:
        print(f"ID: {hit.id}, 距离: {hit.distance:.4f}, 文本: {hit.entity.get('text')}")
        # hit.distance：相似度距离（L2越小越相似，IP越大越相似）

##删除集合
#1.实例化删除
collection = Collection('collection_name')##实例化集合
collection.drop()
#2.直接删除
utility.drop_collection('collecction_name')

