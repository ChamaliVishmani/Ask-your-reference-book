import pypyodbc
import pickle

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


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


def createVectorAndAddToDB(cursor, conn):
    # create vector
    llm = Ollama(model="llama2")
    lease_note_loader = WebBaseLoader(
        "https://drive.google.com/file/d/1grZyR7sXYPY0eiSqh_Yr5-99xPZ7snL_/view?usp=drive_link")
    finance_docs = lease_note_loader.load()

    embeddings = OllamaEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(finance_docs)
    vector = FAISS.from_documents(documents, embeddings)

    # parameters to add to the database
    vector_serialized = pickle.dumps(vector)
    document_id = "lease_note"

    # check if embeddings table exists if not create it
    query = """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'embeddings') CREATE TABLE embeddings (document_id NVARCHAR(255) PRIMARY KEY, embedding NVARCHAR(MAX))"""
    cursor.execute(query)
    conn.commit()

    # add_vector_query = """INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)"""
    add_vector_query = """
    MERGE INTO embeddings AS target
    USING (VALUES (?, ?)) AS source (document_id, embedding)
    ON target.document_id = source.document_id
    WHEN NOT MATCHED THEN
        INSERT (document_id, embedding) VALUES (?,?);

    """
    cursor.execute(add_vector_query, (document_id,
                                      vector_serialized, document_id, vector_serialized))
    conn.commit()


def askQuestion():
    pass


if __name__ == "__main__":
    cursor, conn = databaseConncetion()
    createVectorAndAddToDB(cursor, conn)


# prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
#     <context>
#     {context}
#     <context>

#     Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)
