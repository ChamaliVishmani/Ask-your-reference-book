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

server = "CHAMALI-ASUSVIV\SQLEXPRESS"
database = "ask-ref-db"
driverName = "SQL Server"

connectionStr = f"""DRIVER={{{driverName}}};SERVER={server};DATABASE={database};Trust_Connection=yes;"""
conn = pypyodbc.connect(connectionStr)

if conn:
    print("Connected to the database")
else:
    print("Connection failed")

cursor = conn.cursor()

# check if embeddings table exists if not create it
query = """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'embeddings') CREATE TABLE embeddings (document_id NVARCHAR(255) PRIMARY KEY, embedding NVARCHAR(MAX))"""
cursor.execute(query)
conn.commit()

llm = Ollama(model="llama2")
lease_note_loader = WebBaseLoader(
    "https://drive.google.com/file/d/1grZyR7sXYPY0eiSqh_Yr5-99xPZ7snL_/view?usp=drive_link")
finance_docs = lease_note_loader.load()

embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(finance_docs)
vector = FAISS.from_documents(documents, embeddings)

# add vector to database
vector_serialized = pickle.dumps(vector)
document_id = "lease_note"

add_vector_query = """INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)"""
cursor.execute(add_vector_query, (document_id, vector_serialized))
conn.commit()

prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
    <context>
    {context}
    <context>
    
    Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
