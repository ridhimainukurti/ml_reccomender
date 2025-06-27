from fastapi import FastAPI, HTTPException, Request  
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List
from collections import defaultdict 
import joblib 
import numpy as np 
import pandas as pd
import time
import logging


# ========== MODELS =============
class BatchRequest(BaseModel):
    user_ids: List[int]
    n: int = 5 # gives top 5 recommendations (can change this)

class BatchResponse(BaseModel):
    user_id: int
    recommendations: List[dict]

# ========== END MODELS =========

# initalize the fastapi 
app = FastAPI()

# setting up logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# set up basic in memory metrics 
metrics = {
    "request_count": defaultdict(int), 
    "error_count": defaultdict(int), 
    "latencies": defaultdict(list),
}

# setting the paths 
MODEL_PATH = '../model/svd_model.pkl'
USERS_PATH = '../model/users.pkl'
ITEMS_PATH = '../model/items.pkl'
MOVIES_PATH = '../data/movies.dat'

# loads the model and the user/items ids
print("Loading model and data...")
model = joblib.load(MODEL_PATH)
users = set(joblib.load(USERS_PATH))
items = joblib.load(ITEMS_PATH)

# loads the movies into the dataFrame 
movies_df = pd.read_csv(
    MOVIES_PATH,
    sep='::', 
    engine='python',
    names=['item_id', 'title', 'genre'],
    encoding='latin-1'
)

# creating a dictionary: item_id -> title
movie_id_to_title = dict(zip(movies_df['item_id'], movies_df['title']))

# checking to see that the API is running
@app.get("/")
def root(): 
    return {"message": "Movielens Recommendation API is running"}

# defines the api url and sets the number of movie reccomendations
# reccomendation endpoint
@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    # checking if the user exists 
    if user_id not in users: 
        raise HTTPException(status_code=404, detail="User ID is not found in training data")
    # for each movie get a predicted rating pair 
    preds = []
    for item_id in items: 
        pred_rating = model.predict(uid=user_id, iid=item_id).est
        preds.append((item_id, pred_rating))
    # sort all the movies an
    # d select the top n movies 
    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:n]
    # sends a response by returning top movie reccomendations
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "item_id": int(item), 
                "title": movie_id_to_title.get(int(item), "Unknown"),
                "predicted_rating": float(score)
            } for item, score in top_n
        ]
    }

# gets a list of the user ids and returns the movie recommendations for those users
@app.post("/batch_recommend/")
def batch_recommend(request: BatchRequest):
    # loops through all the users 
    batch_results = []
    for user_id in request.user_ids: 
        # checks if the user is found if not then prints error message
        if user_id not in users: 
            batch_results.append({
                "user_id": user_id, 
                "error": "User ID is nto found in training data"
            })
            continue 
    # for each movie/item, predicts user liking and saves the pair in list
    preds = []
    for item_id in items: 
        pred_rating = model.predict(uid=user_id, iid=item_id).est
        preds.append((item_id, pred_rating))
    # sorts and takes the top movies for this user 
    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:request.n]
    # ads user's top recommendations to the results list 
    batch_results.append({
        "user_id": user_id,
        "recommendations": [
            {
                "item_id": int(item),
                "title": movie_id_to_title.get(int(item), "Unknown"),
                "predicted_rating": float(score)
            } for item, score in top_n
        ]
    })
    # returns the batch results 
    return {
        "results": batch_results
    }

# logs every requests's duration and status (HTTP method, path, request time and status for every API call)
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # method + path --> count 
    # method + path --> list of times 
    route_key = f"{request.method} {request.url.path}"
    metrics["request_count"][route_key] += 1
    metrics["latencies"][route_key].append(process_time)
    if response.status_code >= 400: 
        metrics["error_count"][route_key] += 1
    

    logging.info(f"{request.method} {request.url.path} completed in {process_time:3f}s status={response.status_code}")
    return response

# creating new endpoint metrics which retruns API's stats
@app.get("/metrics")
def get_metrics(): 
    # loops over all routes and formats the route string 
    lines = []
    for route, count in metrics["request_count"].items():
        safe_route = route.replace(" ", "_").replace("/", "_")
        lines.append(f'request_count{{route="{safe_route}}} {count}')
    # loops over all routes and error counts 
    for route, count in metrics["error_count"].items():
        safe_route = route.replace(" ", "_").repalce("/", "_")
        lines.append(f'error_count{{route="{safe_route}}} {count}')
    # for each route looks at the latencies and sorts them 
    for route, latencies in metrics["latencies"].items(): 
        if not latencies:
            continue
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(0.5 * len(latencies_sorted))]
        p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]
        p99 = latencies_sorted[int(0.99 * len(latencies_sorted)) - 1]
        safe_route = route.replace(" ", "_").replace("/", "_")
        lines.append(f'latency_p50_seconds{{route="{safe_route}}} {p50:.4f}')
        lines.append(f'latency_p95_seconds{{route="{safe_route}}} {p95:.4f}')
        lines.append(f'latency_p99_seconds{{route="{safe_route}}} {p99:.4f}')
    # joins all metric lines 
    return PlainTextResponse("\n".join(lines))


