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


def createRAGChain(chromaDB, llm):
    retriever = chromaDB.as_retriever(
        search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | formatDocs, "question": RunnablePassthrough()
         } | prompt | llm
    )
    return rag_chain


def askQuestion(question, rag_chain):
    return rag_chain.invoke(question)


def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


file_path = r"pdfs\03_Leases.pdf"
llm = createLLM()
chromaDB, docs = addPDFToChroma(file_path)
rag_chain = createRAGChain(chromaDB, llm)

question = "What is the definition of a lease?"
answer = askQuestion(question, rag_chain)
print(answer)