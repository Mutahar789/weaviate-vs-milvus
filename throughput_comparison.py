import time
import threading
import statistics
import shutil
import matplotlib.pyplot as plt
import os

import data
import vectorstore
import queries

if not os.path.exists("./results/plots"):
    os.makedirs("./results/plots")

documents = data.load_data(0)
texts = data.splitter.transform_documents(documents)

latencies = []

def search(vs, query):
    global latencies
    _, search_time = vs.search(query)
    latencies.append(search_time)

def mean_latency(qps, vs):
    global latencies
    latencies = []
    threads = []

    wait = 1/qps
    for q in queries.queries:
        t = threading.Thread(target=search, args=(vs, q))
        t.start()
        threads.append(t)
        time.sleep(wait)
    
    for t in threads:
        t.join()
        
    return statistics.mean(latencies)

vector_databases = {
    "weaviate": vectorstore.weaviate,
    "milvus": vectorstore.milvus
}

throughput_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
mean_latencies = {
    "weaviate": [],
    "milvus": []
}
for name, cls in vector_databases.items():
    vs = cls()
    vs.index(texts)
    time.sleep(1)
    for qps in throughput_values:
        mean_latencies[name].append(mean_latency(qps, vs))
    vs.stop()

plt.figure(figsize=(12, 8))
plt.plot(throughput_values, mean_latencies['milvus'], color='red', marker='o', label='Milvus')
plt.plot(throughput_values, mean_latencies["weaviate"], color='blue', marker='x', label='Weaviate')
plt.xlabel('queries per second')
plt.ylabel('mean latency (seconds)')
title_props = {'weight': 'bold', 'size': 16}
plt.title('Throughput vs. mean latency', fontdict=title_props)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.xticks(throughput_values)
plt.savefig(f"./results/plots/throughput_vs_latency")

data.clean_up()
shutil.rmtree("./__pycache__")
