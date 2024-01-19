from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

model = {
    "Paul": "Oui"
}


def compute_model_result(parameters):
    # Load the model from the file
    # Fit the model under the parameters from the route
    # Return the model resuls
    pass


@app.get("/models-results")
def get_models_results(id: str) -> str:
    #compute_model_result(parameters=id)
    return model.get(id, "Aucun")