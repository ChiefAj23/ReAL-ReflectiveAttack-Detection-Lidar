import pandas as pd
import time
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

model_filename = 'svm_model_ex1_gamma_100.pkl'
loaded_model = joblib.load(model_filename)
df2 = pd.concat([pd.read_csv('nvidia_6_n_inference.csv'), pd.read_csv('nvidia_6_s_inference.csv')], ignore_index=True)
df2 = df2.dropna(subset=['object'])
X2 = df2[['angle', 'distance']]
y2 = df2['object']
start_time = time.time()
y2_pred = loaded_model.predict(X2)
end_time = time.time()
inference_latency = end_time - start_time
inference_accuracy = accuracy_score(y2, y2_pred)
inference_f1 = f1_score(y2, y2_pred, average='weighted')
print(f'Inference Accuracy: {inference_accuracy:.4f}')
print(f'Inference F1 Score: {inference_f1:.4f}')
print(f'Inference Latency: {inference_latency:.4f} seconds')
print('Classification Report for Inference Data:\n', classification_report(y2, y2_pred))
