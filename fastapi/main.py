from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI is working!"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

@app.get("/search")
def search(q: str, limit: int = 5):
    return {"query": q, "limit": limit}

class Item(BaseModel):
    name : str
    price: float
    in_stock: bool

@app.post("/items")
def create_item(item: Item):
    return {
        "name": item.name,
        "price": item.price
    }

@app.get("/slow")
async def slow_api():
    return {"message": "async endpoint"}

class Prompt(BaseModel):
    text: str
@app.post("/generate")
def generate_text(prompt: Prompt):

    # fake LLM response
    response = f"Generated response for: {prompt.text}"
    return {"response": response}

@app.post("/rag")
def rag(query: str):
    docs = vectordb.search(query)
    answer = llm.generate(context = docs, question = query)
    return {"answer": answer}