from pymilvus import connections,FieldSchema,CollectionSchema,Collection,utility,DataType
from langchain_community.document_loaders import Docx2txtLoader,PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import Config
import pandas as pd

'''
RecursiveCharacterTextSplitter.split_documents()-->用于处理加载的文档数据，结果列表需要通过chunk.page_content取出数据
RecursiveCharacterTextSplitter.split_text()-->用于处理纯文本数据，返回存文本结果列表
'''


class Documentprocess:

    '转码'
    @staticmethod
    def encoder(text_chunk):
        return SentenceTransformer(model_name_or_path='./all-MiniLM-L6-v2').encode(text_chunk,normalize_embeddings=True).tolist()

    '加载文件'
    @staticmethod
    def doc_load(file_path:str):

        if file_path.endswith('pdf'):
            load = PDFMinerLoader(file_path)
        elif file_path.endswith('doc')|file_path.endswith('docx'):
            load = Docx2txtLoader(file_path)
        else:
            raise ValueError('仅支持pdf和doc文件')
        
        return load.load()
    
    '分割文件'
    @staticmethod
    def doc_split(doc):

        doc_split = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap=50,
            length_function=len
        )

        return doc_split.split_documents(doc)


class mymilvus():
    '链接'
    def __init__(self):
        
        connections.connect(alias='default',host=Config.MILVUS_HOST,port=Config.MILVUS_PORT)
        if connections.has_connection(alias='default'):
            print('成功连接')

    '查询和创建'
    def create(self,collection_name):

        if utility.has_collection(collection_name):
            print('数据库中存在相应的表')
            collection = Collection(Config.COLLECTION_NAME)
            
            if collection.has_index():

                print('表中embedding存在索引，无需创建')
            
            else:
                print('为该表创建索引')
                collection.create_index(
                    field_name='embedding',
                    index_params={
                            'index_type':"IVF_FLAT",
                            'metric_type':'L2',
                            "params": {"nlist": 128}
                    }
                )
 
        else:
            print('数据库中不存在相应的表，现在开始创建')

            fields = [
                FieldSchema(name='id',dtype=DataType.INT64,is_primary=True,auto_id=True),
                FieldSchema(name='embedding',dtype=DataType.FLOAT_VECTOR,dim=Config.DIMENSION),
                FieldSchema(name='text',dtype=DataType.VARCHAR,max_length=200)
            ]
            try:
                collection = Collection(name=collection_name,schema=CollectionSchema(fields=fields,description='id、embdding、text'))
                if collection:
                    print(f'{collection_name}创建完成,创建索引')
                    collection.create_index(
                        field_name='embedding',
                        index_params={
                            'index_type':"IVF_FLAT",
                            'metric_type':'L2',
                            "params": {"nlist": 128}
                        }
                    )
                    
            except Exception as e:
                print(f'创建失败',e)

        return collection
    

    '插入'
    def insert_values(self,chunk_text_list):

        texts = [doc.page_content for doc in chunk_text_list]
        embeddings = Documentprocess.encoder(texts)

        collection = self.create(Config.COLLECTION_NAME)
        collection.load()
        '''数据的顺序是embedding/text'''
        insert_result = collection.insert(data=[embeddings,texts])
        collection.flush()
        print(f'成功插入{insert_result.insert_count}条数据')

        return insert_result

    '搜索'
    def similar_search(self,question):

        question = Documentprocess.encoder(question)

        collection = Collection(Config.COLLECTION_NAME)
        collection.load()

        search_params = {
            'metric_type':'L2'
        }
        result = collection.search(
            data=[question],
            anns_field='embedding',
            output_fields=['text'],
            param=search_params,
            limit=5
        )

        similar_result = []
        for chunk in result[0]:
            similar_result.append(chunk.entity.get('text'))


        return similar_result
    

if __name__=='__main__':

    # file = Documentprocess.doc_load('./手机信息.docx')
    # chunk_list = Documentprocess.doc_split(file)
    # chunk_code = [Documentprocess.encoder(chunk) for chunk in chunk_list]
    milvus = mymilvus()
    
    # insert_result = milvus.insert_values(chunk_list)

    search_result = milvus.similar_search(question='cpu')
    print(search_result)

    
