from langchain.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()


def addPDFToChroma(file_path):
    if not os.path.isfile(file_path):
        raise ValueError("File path %s does not exist" % file_path)

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("File path %s is not a valid PDF file" % file_path)

    loader = PyPDFLoader(file_path)
    data = loader.load()

    # split to chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    embedding_func = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    # load to chroma
    chromaDB = Chroma.from_documents(docs, embedding_func)
    return chromaDB, docs


def createLLM():
    gemma_repo_id = "google/gemma-2b-it"

    llm = HuggingFaceEndpoint(repo_id=gemma_repo_id,
                              max_length=1024, temperature=0.1)
    return llm


def createRAGChain():
    retriever = chromaDB.as_retriever(
        search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | formatDocs, "question": RunnablePassthrough()
         } | prompt | llm
    )
    print(rag_chain.invoke("what are the two types of leases?"))


def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


file_path = r"pdfs\03_Leases.pdf"
llm = createLLM()
chromaDB, docs = addPDFToChroma(file_path)
createRAGChain()

# next : https://colab.research.google.com/drive/1JCeL1d6dyC2MrtLI45aII9kIF9m6eOPe#scrollTo=-jZSd_oOCjP5
# TODO - persistent storage for chromaDB
