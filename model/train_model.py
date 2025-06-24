import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os

# reads the ratings file and generates a table called rating (row = user, item, rating)
DATA_PATH = os.path.join('data', 'ratings.dat')
print("Loading data...")
ratings = pd.read_csv(DATA_PATH, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])

# keeping necessary columns 
ratings = ratings[['user', 'item', 'rating']]

# prepares data for model (telling about the format and range of the data)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings, reader)

# test size = 20% of data is for testing, random_state fixes the randomness it uses the exact same split into train/test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# SVD uses a matrix factorization algorithm that breaks it into patterns (guesses what users havent rated yet but might like)
print("Training SVD model...")
algo = SVD(n_factors=50, n_epochs=20, random_state=42)
algo.fit(trainset)

# tell us how far off our predicted ratings are from the real ratings
predictions = algo.test(testset)
rmse = sum([(true_r - est_r) ** 2 for (_, _, true_r, est_r, _) in predictions]) / len(predictions)
print(f"Test RMSE: {rmse:4f}")

# saved trained model so you we can load later 
MODEL_PATH = os.path.join("svd_model.pkl")
joblib.dump(algo, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# saving users and item lists for later esp since we need to know which users/items the model has seen during training (predictions)
users = ratings['user'].unique()
items = ratings['item'].unique()
joblib.dump(users, os.path.join('users.pkl'))
joblib.dump(items, os.path.join('items.pkl'))
print("User and item lists saved for inference")




