import statistics
import time
import shutil

import data
import vectorstore
import VizLogger
import queries

for itr in range(len(data.datasizes)):
    documents = data.load_data(itr)
    texts = data.splitter.transform_documents(documents)
    num_docs = data.datasizes[itr]
    num_vectors = len(texts)
    
    vector_databases = {
        "weaviate": vectorstore.weaviate,
        "milvus": vectorstore.milvus
    }
    results = {
        "build_time": {},
        "search_time": {}
    }

    for name, cls in vector_databases.items():
        build_times = []
        search_times = []
        for i in range(5):
            vs = cls()
            build_time = vs.index(texts)
            time.sleep(1)
            build_times.append(build_time)
            for j in range(10):
                _, search_time = vs.search(queries.queries[j])
                search_times.append(search_time)
            vs.stop()

        results["build_time"][name] = statistics.mean(build_times)
        results["search_time"][name] = statistics.mean(search_times)

    VizLogger.log_data(num_docs, num_vectors, results)

    data.clean_up()

VizLogger.generate_plots()
shutil.rmtree("./__pycache__")

