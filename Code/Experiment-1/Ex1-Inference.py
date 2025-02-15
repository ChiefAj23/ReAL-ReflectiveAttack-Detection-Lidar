import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

model = joblib.load("/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/svm_model_ex1_gamma_100.pkl")
inference_n = pd.read_csv('/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_n_inference.csv').dropna(subset=['object'])
inference_s = pd.read_csv('/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_s_inference.csv').dropna(subset=['object'])
combined_inference = pd.concat([inference_n, inference_s], ignore_index=True)
X_combined = combined_inference[['angle', 'distance (mm)']]
y_combined = combined_inference['object']
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
start_time = time.time()
y_pred_combined = model.predict(X_combined)
latency = time.time() - start_time
accuracy = accuracy_score(y_combined, y_pred_combined)
f1 = f1_score(y_combined, y_pred_combined, average='weighted')
print("Results for the combined inference dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Latency: {latency:.4f} seconds")