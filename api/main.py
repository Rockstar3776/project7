from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

model = {
    "paul": "Oui"
}

@app.get("/models-results")
def read_root(id: str):
    return model[id]