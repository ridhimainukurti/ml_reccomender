from fastapi import FastAPI, HTTPException 
import joblib 
import numpy as np 

# initalize the fastapi 
app = FastAPI()

# setting the paths 
MODEL_PATH = '../model/svd_model.pkl'
USERS_PATH = '../model/users.pkl'
ITEMS_PATH = '../model/items.pkl'

# loads the model and the user/items ids
print("Loading model and data...")
model = joblib.load(MODEL_PATH)
users = set(joblib.load(USERS_PATH))
items = joblib.load(ITEMS_PATH)

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
        "recommendations": [{"item_id": int(item), "predicted_rating": float(score)} for item, score in top_n]
    }
    


