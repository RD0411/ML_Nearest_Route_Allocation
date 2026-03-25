# generate_dataset.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import math
import os

# ==============================
# CONFIGURATION
# ==============================
SEED = 42
NUM_PROVIDERS = 5000
NUM_REQUESTS = 50000
MIN_CANDIDATES = 10
MAX_CANDIDATES = 30

OUTPUT_DIR = "data"

SERVICE_TYPES = [
    "TV repair",
    "TV installation",
    "Fridge repair",
    "Fridge installation"
]

# India bounding box (realistic geo distribution)
LAT_MIN, LAT_MAX = 8.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0

np.random.seed(SEED)
random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# HELPER FUNCTIONS
# ==============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def random_datetime(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ==============================
# GENERATE PROVIDERS
# ==============================
providers = []

for i in range(NUM_PROVIDERS):
    provider_id = f"P{i+1}"

    skills = random.sample(SERVICE_TYPES, k=random.randint(1, 3))

    providers.append({
        "provider_id": provider_id,
        "provider_lat": np.random.uniform(LAT_MIN, LAT_MAX),
        "provider_lon": np.random.uniform(LON_MIN, LON_MAX),
        "skills": "|".join(skills),
        "rating": round(np.random.uniform(3.0, 5.0), 2),
        "acceptance_rate_30d": round(np.random.uniform(0.6, 1.0), 2),
        "avg_response_time_min": round(np.random.uniform(2, 30), 2),
        "completed_jobs_90d": np.random.randint(10, 300)
    })

providers_df = pd.DataFrame(providers)
providers_df.to_csv(f"{OUTPUT_DIR}/providers.csv", index=False)


# ==============================
# GENERATE REQUESTS
# ==============================
start_time = datetime(2024, 1, 1)
end_time = datetime(2025, 12, 31)

requests = []

for i in range(NUM_REQUESTS):
    requests.append({
        "request_id": f"R{i+1}",
        "request_time": random_datetime(start_time, end_time),
        "service_type": random.choice(SERVICE_TYPES),
        "client_lat": np.random.uniform(LAT_MIN, LAT_MAX),
        "client_lon": np.random.uniform(LON_MIN, LON_MAX)
    })

requests_df = pd.DataFrame(requests)
requests_df = requests_df.sort_values("request_time")
requests_df.to_csv(f"{OUTPUT_DIR}/requests.csv", index=False)


# ==============================
# GENERATE CANDIDATE ALLOCATIONS
# ==============================
candidate_rows = []

for _, req in requests_df.iterrows():

    num_candidates = random.randint(MIN_CANDIDATES, MAX_CANDIDATES)
    sampled_providers = providers_df.sample(num_candidates)

    best_score = -1
    best_index = None
    local_rows = []

    for _, prov in sampled_providers.iterrows():

        is_available = np.random.choice([0, 1], p=[0.2, 0.8])

        has_skill_match = int(req["service_type"] in prov["skills"].split("|"))

        distance = haversine(
            req["client_lat"], req["client_lon"],
            prov["provider_lat"], prov["provider_lon"]
        )

        # scoring logic
        score = 0
        if is_available and has_skill_match:
            score = (prov["rating"] * 2) + (1 / (distance + 1))

        row = {
            "request_id": req["request_id"],
            "request_time": req["request_time"],
            "service_type": req["service_type"],
            "client_lat": req["client_lat"],
            "client_lon": req["client_lon"],
            "provider_id": prov["provider_id"],
            "provider_lat": prov["provider_lat"],
            "provider_lon": prov["provider_lon"],
            "is_available": is_available,
            "has_skill_match": has_skill_match,
            "provider_rating": prov["rating"],
            "distance_km": round(distance, 2),
            "acceptance_rate_30d": prov["acceptance_rate_30d"],
            "avg_response_time_min": prov["avg_response_time_min"],
            "completed_jobs_90d": prov["completed_jobs_90d"],
            "selected_label": 0
        }

        local_rows.append(row)

        if score > best_score:
            best_score = score
            best_index = len(local_rows) - 1

    # assign label
    if best_score > 0:
        local_rows[best_index]["selected_label"] = 1

    candidate_rows.extend(local_rows)

candidate_df = pd.DataFrame(candidate_rows)
candidate_df.to_csv(f"{OUTPUT_DIR}/candidate_allocations.csv", index=False)


# ==============================
# TIME-BASED SPLIT
# ==============================
candidate_df = candidate_df.sort_values("request_time")

unique_requests = candidate_df["request_id"].drop_duplicates().values

train_cut = int(0.7 * len(unique_requests))
valid_cut = int(0.85 * len(unique_requests))

train_ids = set(unique_requests[:train_cut])
valid_ids = set(unique_requests[train_cut:valid_cut])
test_ids = set(unique_requests[valid_cut:])

train_df = candidate_df[candidate_df["request_id"].isin(train_ids)]
valid_df = candidate_df[candidate_df["request_id"].isin(valid_ids)]
test_df = candidate_df[candidate_df["request_id"].isin(test_ids)]

train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
valid_df.to_csv(f"{OUTPUT_DIR}/valid.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)


# ==============================
# DATA DICTIONARY
# ==============================
data_dict = [
    ["request_id", "Unique request identifier"],
    ["request_time", "Timestamp of request"],
    ["service_type", "Type of service requested"],
    ["client_lat", "Client latitude"],
    ["client_lon", "Client longitude"],
    ["provider_id", "Provider identifier"],
    ["provider_lat", "Provider latitude"],
    ["provider_lon", "Provider longitude"],
    ["is_available", "Provider availability (0/1)"],
    ["has_skill_match", "Skill match (0/1)"],
    ["provider_rating", "Rating of provider"],
    ["distance_km", "Distance between client and provider"],
    ["acceptance_rate_30d", "Acceptance rate in last 30 days"],
    ["avg_response_time_min", "Average response time"],
    ["completed_jobs_90d", "Completed jobs in 90 days"],
    ["selected_label", "1 if selected, else 0"]
]

pd.DataFrame(data_dict, columns=["column", "description"]) \
    .to_csv(f"{OUTPUT_DIR}/data_dictionary.csv", index=False)


# ==============================
# DATA QUALITY CHECKS
# ==============================
print("\n==== DATA QUALITY CHECKS ====")

assert providers_df["provider_id"].isnull().sum() == 0
assert requests_df["request_id"].isnull().sum() == 0

assert candidate_df["client_lat"].between(-90, 90).all()
assert candidate_df["client_lon"].between(-180, 180).all()

# label consistency
label_check = candidate_df.groupby("request_id")["selected_label"].sum()
assert (label_check <= 1).all()

print("All checks passed!")

# ==============================
# SUMMARY
# ==============================
print("\n==== SUMMARY ====")
print(f"Providers: {len(providers_df)}")
print(f"Requests: {len(requests_df)}")
print(f"Candidates: {len(candidate_df)}")

print("\nSplit Sizes:")
print(f"Train: {len(train_df)}")
print(f"Valid: {len(valid_df)}")
print(f"Test: {len(test_df)}")

positive_rate = candidate_df["selected_label"].mean()
print(f"\nPositive Rate: {round(positive_rate, 4)}")

print("\nDataset generated successfully in 'data/' folder.")