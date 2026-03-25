from fastapi import FastAPI
from app.schema import RequestData
from app.predictor import predict_best_provider

app = FastAPI(title="FixItPro Allocation API")


@app.get("/")
def home():
    return {"message": "FixItPro API is running 🚀"}


@app.post("/allocate")
def allocate(request: RequestData):

    best = predict_best_provider(request)

    if best is None:
        return {"message": "No provider available"}

    return {
        "provider_id": best["provider_id"],
        "score": float(best["score"]),
        "rating": float(best["provider_rating"]),
        "distance_km": float(best["distance_km"])
    }