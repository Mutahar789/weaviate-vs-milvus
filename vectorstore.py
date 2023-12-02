import time
import subprocess

from langchain.schema.document import Document
from langchain.vectorstores import Milvus
from langchain.vectorstores import Weaviate
from weaviate import Client

import embeddings

class vectorstore:
    def __init__(self):
        self.vectorstore = None

    def index(self, documents):
        start = time.time()
        self.vectorstore.add_documents(documents)
        end = time.time()
        return end-start
    
    def search(self, query):
        start = time.time()
        docs = self.vectorstore.similarity_search(query, k=10)
        end = time.time()
        return docs, end-start

class milvus(vectorstore):
    def __init__(self):
        # start container
        subprocess.run(["bash", "./milvus/run.sh"])
        time.sleep(5)

        # create a collection with HNSW index
        milvus = Milvus(
            embedding_function= embeddings.embeddings_model,
            collection_name= "benchmark",
            connection_args={"host": "127.0.0.1", "port": "19530"},
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {
                    "M": 32,
                    "efConstruction": 128,
                    "ef": 64
                },
            }
        )
        init = [Document(page_content="", metadata="")]
        self.vectorstore = milvus.from_documents(init, embedding=embeddings.embeddings_model)
    
    def stop(self):
        subprocess.run(["bash", "./milvus/stop.sh"])

class weaviate(vectorstore):
    def __init__(self):
        # start container
        subprocess.run(["bash", "./weaviate/run.sh"])
        time.sleep(5)
        
        # create a class with HNSW index
        client = Client(
            url="http://localhost:8080",
            timeout_config=(5, 15)
        )
        class_obj = {
            'class': 'Benchmark',
            'properties': [
                {
                    'name': 'text',
                    'dataType': ['text'],
                },
            ],
            'vectorIndexConfig': {
                'skip': False,
                'maxConnections': 32,
                'efConstruction': 128,
                'ef': 64,
                'distance': 'dot',
            },
            'vectorIndexType': 'hnsw',
        }

        client.schema.create_class(class_obj)
        init = [Document(page_content="", metadata="")]
        self.vectorstore = Weaviate.from_documents(
            init,
            embeddings.embeddings_model,
            weaviate_url="http://localhost:8080",
            index_name = "Benchmark"
        )

    def stop(self):
        subprocess.run(["bash", "./weaviate/stop.sh"])
