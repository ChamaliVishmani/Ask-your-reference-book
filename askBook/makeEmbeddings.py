import pypyodbc
import pickle

import chromadb
import os

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


def createChromaDB():
    chroma_client = chromadb.PersistentClient("chromaDB")

    collection = chroma_client.create_collection(name="reference_collection")


def addPDFtoChroma(file_path):
    chroma_client = chromadb.PersistentClient("chromaDB")

    collection = chroma_client.get_collection(name="reference_collection")

    # file_path = r"pdfs\03_Leases.pdf"

    if not os.path.isfile(file_path):
        raise ValueError("File path %s does not exist" % file_path)

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("File path %s is not a valid PDF file" % file_path)

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    for page in pages:
        page_id = "id3" + str(page.metadata["page"])
        # embedding done by chroma
        collection.add(documents=[page.page_content], ids=[page_id])

    results = collection.query(
        query_texts=["what are the two types of leases?"], n_results=2)
    print("results ", results)

    print("Added pdf to chromaDB")


'''
def databaseConncetion():
    server_name = "CHAMALI-ASUSVIV\SQLEXPRESS"
    database_name = "ask-ref-db"
    driver_name = "SQL Server"

    connection_string = f"""DRIVER={{{driver_name}}};SERVER={server_name};DATABASE={database_name};Trust_Connection=yes;"""
    conn = pypyodbc.connect(connection_string)

    if conn:
        print("Connected to the database")
    else:
        print("Connection failed")

    cursor = conn.cursor()

    return cursor, conn

def createVectorAndAddToDB():
    cursor, conn = databaseConncetion()

    # create vector
    lease_note_loader = PyPDFLoader("pdfs/03_Leases.pdf")
    finance_pages = lease_note_loader.load_and_split()
    # print("finance_docs ", finance_pages[3])
    embeddings = OllamaEmbeddings()

    vector = FAISS.from_documents(finance_pages, embeddings)
    print("vector ", vector)

    # parameters to add to the database
    vector_serialized = pickle.dumps(
        vector)

    document_id = "lease_note"

    # check if embeddings table exists if not create it
    query = """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'vector_embeddings') CREATE TABLE vector_embeddings (document_id NVARCHAR(255) PRIMARY KEY, embedding VARBINARY(MAX))"""
    cursor.execute(query)
    conn.commit()

    add_vector_query = """
    MERGE INTO vector_embeddings AS target
    USING (VALUES (?, ?)) AS source (document_id, embedding)
    ON target.document_id = source.document_id
    WHEN MATCHED THEN
        UPDATE SET embedding = source.embedding
    WHEN NOT MATCHED THEN
        INSERT (document_id, embedding) VALUES (?,?);
    """
    print("vector_serialized ", type(vector_serialized))

    cursor.execute(add_vector_query, (document_id,
                                      vector, document_id, vector))
    conn.commit()

    conn.close()

    return vector

def getVectorFromDB(document_id):
    cursor, conn = databaseConncetion()

    retrieve_query = "SELECT * FROM vector_embeddings WHERE document_id = ?"
    cursor.execute(retrieve_query, (document_id,))
    row = cursor.fetchone()
    print("row ", row)
    print("type ", type(bytes(row[1], 'utf-8')))

    if row:
        vector_serialized = bytes(row[1], 'latin1')

        vector = pickle.loads(
            vector_serialized)

        print("vector ", vector)
    else:
        print("No vector found for document_id ", document_id)

    conn.close()
'''


def askQuestion():
    llm = Ollama(model="llama2")
    prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
    <context>
    {context}
    <context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    vector = getVectorFromDB("lease_note")

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke(
        {"input": "What is a lease and what are the two types of leases?"})
    print(response)


if __name__ == "__main__":

    # vector = createVectorAndAddToDB()
    # askQuestion()
    file_path = r"pdfs\03_Leases.pdf"
    addPDFtoChroma(file_path)
