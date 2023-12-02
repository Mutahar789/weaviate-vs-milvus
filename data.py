import shutil
import os

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import SentenceTransformersTokenTextSplitter

datasizes = [3, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def create_data(itr):
    os.mkdir("./temp")
    for i in range(datasizes[itr]):
        filename = f'Resume_{i}.pdf'
        shutil.copyfile(f"./Resumes/{filename}", f"./temp/{filename}")

directory_loader = DirectoryLoader(
    "./temp/",
    glob="*.pdf",
    use_multithreading=True,
    loader_cls=PyPDFLoader,
    show_progress=True
)

def load_data(itr):
    create_data(itr)
    documents = directory_loader.load()
    return documents

def clean_up():
    shutil.rmtree('./temp')

splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
