import pandas as pd
import numpy as np
import joblib
import random
import math

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model.pkl")

providers_df = pd.read_csv("data/providers.csv")

# ==============================
# HAVERSINE FUNCTION
# ==============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==============================
# GENERATE NEW REQUEST
# ==============================
def generate_request():
    return {
        "service_type": random.choice([
            "TV repair",
            "TV installation",
            "Fridge repair",
            "Fridge installation"
        ]),
        "client_lat": np.random.uniform(8.0, 37.0),
        "client_lon": np.random.uniform(68.0, 97.0)
    }

# ==============================
# BUILD CANDIDATES
# ==============================
def get_candidates(request, num_candidates=20):
    sampled = providers_df.sample(num_candidates)

    rows = []

    for _, prov in sampled.iterrows():

        is_available = np.random.choice([0, 1], p=[0.2, 0.8])
        has_skill_match = int(request["service_type"] in prov["skills"].split("|"))

        distance = haversine(
            request["client_lat"], request["client_lon"],
            prov["provider_lat"], prov["provider_lon"]
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

    # filter valid providers (important)
    valid_candidates = candidates[
        (candidates["is_available"] == 1) &
        (candidates["has_skill_match"] == 1)
    ]

    if len(valid_candidates) == 0:
        return None, candidates

    best = valid_candidates.sort_values("score", ascending=False).iloc[0]

    return best, candidates


# ==============================
# RUN TEST
# ==============================
if __name__ == "__main__":

    request = generate_request()

    best, candidates = predict_best_provider(request)

    print("\n===== NEW REQUEST =====")
    print(request)

    print("\n===== TOP 5 CANDIDATES =====")
    print(candidates.sort_values("score", ascending=False).head())

    if best is not None:
        print("\n✅ BEST PROVIDER SELECTED:")
        print(best)
    else:
        print("\n⚠️ No valid provider found")