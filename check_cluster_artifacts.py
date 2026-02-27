import json, joblib
import numpy as np

scaler = joblib.load("cluster_scaler.pkl")
kmeans = joblib.load("kmeans_k6.pkl")
feats = json.load(open("cluster_feature_list.json"))

print("features:", feats)
print("n_features:", len(feats))

print("scaler mean shape:", scaler.mean_.shape)
print("kmeans centers shape:", kmeans.cluster_centers_.shape)

# smoke test prediction
x = np.zeros((1, len(feats)))
print("predict:", kmeans.predict(x))
