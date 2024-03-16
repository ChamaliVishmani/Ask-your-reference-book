# start server -> uvicorn interactAPIs:app --reload
from fastapi import FastAPI
from pydantic import BaseModel

from makeEmbeddings import askQuestion


class Question(BaseModel):
    question: str
    chapter: int


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/ask")
async def ask(question: Question):
    return askQuestion(question.question)
