import pandas as pd
import numpy as np
import joblib
from app.utils import haversine

# ==============================
# LOAD MODEL + DATA
# ==============================
model = joblib.load("models/model.pkl")
providers_df = pd.read_csv("data/providers.csv")


# ==============================
# BUILD CANDIDATES
# ==============================
def get_candidates(request, num_candidates=20):

    # calculate distance for all providers
    providers_with_distance = providers_df.copy()
    providers_with_distance["distance_km"] = providers_with_distance.apply(
        lambda x: haversine(
            request.client_lat,
            request.client_lon,
            x["provider_lat"],
            x["provider_lon"]
        ),
        axis=1
    )

    # take nearest providers
    sampled = providers_with_distance.nsmallest(num_candidates, "distance_km")

    rows = []

    for _, prov in sampled.iterrows():

        is_available = np.random.choice([0, 1], p=[0.2, 0.8])
        has_skill_match = int(request.service_type in prov["skills"].split("|"))

        distance = haversine(
            request.client_lat,
            request.client_lon,
            prov["provider_lat"],
            prov["provider_lon"]
        )

        rows.append({
            "provider_id": prov["provider_id"],
            "is_available": is_available,
            "has_skill_match": has_skill_match,
            "provider_rating": prov["rating"],
            "distance_km": distance,
            "acceptance_rate_30d": prov["acceptance_rate_30d"],
            "avg_response_time_min": prov["avg_response_time_min"],
            "completed_jobs_90d": prov["completed_jobs_90d"]
        })

    return pd.DataFrame(rows)


# ==============================
# PREDICT BEST PROVIDER
# ==============================
def predict_best_provider(request):

    candidates = get_candidates(request)

    features = [
        "is_available",
        "has_skill_match",
        "provider_rating",
        "distance_km",
        "acceptance_rate_30d",
        "avg_response_time_min",
        "completed_jobs_90d"
    ]

    candidates["score"] = model.predict(candidates[features])

    # filter valid providers
    valid = candidates[
        (candidates["is_available"] == 1) &
        (candidates["has_skill_match"] == 1)
    ]

    if len(valid) == 0:
        return None

    best = valid.sort_values("score", ascending=False).iloc[0]

    return best