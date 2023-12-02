import csv
import os
import shutil
import matplotlib.pyplot as plt


shutil.rmtree("./results", ignore_errors=True)
os.mkdir("./results")
os.mkdir("./results/raw_data")
os.mkdir("./results/plots")

csv_files = ["build_time", "search_time"]

for csv_file in csv_files:
    with open(f"./results/raw_data/{csv_file}.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["number of docs", "number of vectors", "weaviate", "milvus"])


def log_data(num_docs, num_vectors, results):
    for csv_file in csv_files:
        with open(f"./results/raw_data/{csv_file}.csv", mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([num_docs, num_vectors, results[csv_file]["weaviate"], results[csv_file]["milvus"]])


def generate_plots():
    for csv_file in csv_files:
        data = []
        with open(f"./results/raw_data/{csv_file}.csv", mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                data.append(row)

        # Extract data columns
        no_of_docs = [int(row[0]) for row in data]
        milvus_times = [float(row[3]) for row in data]
        weaviate_times = [float(row[2]) for row in data]

        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.plot(no_of_docs, milvus_times, color='red', marker='o', label='Milvus')
        plt.plot(no_of_docs, weaviate_times, color='blue', marker='x', label='Weaviate')
        plt.xlabel('no. of Docs')
        plt.ylabel(f'{csv_file} (seconds)')
        title_props = {'weight': 'bold', 'size': 16}
        plt.title('Comparison of Weaviate and Milvus', fontdict=title_props)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.xticks(no_of_docs)
        plt.savefig(f"./results/plots/{csv_file}")
