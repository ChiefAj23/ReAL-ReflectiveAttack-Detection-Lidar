import pandas as pd
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

model_object_1_path = '/home/orin1/Documents/Luke/ex4/ex4_final_new_approach/model/object_1/svm_model_object_1.pkl'
model_object_2_path = '/home/orin1/Documents/Luke/ex4/ex4_final_new_approach/model/object_2/svm_model_object_2.pkl'

model_object_1 = joblib.load(model_object_1_path)
model_object_2 = joblib.load(model_object_2_path)
file_1 = '/home/orin1/Documents/Luke/ex4/battery_s_tissue_s.csv'
file_2 = '/home/orin1/Documents/Luke/ex4/battery_n_tissue_s.csv'
file_3 = '/home/orin1/Documents/Luke/ex4/battery_s_tissue_n.csv'
file_4 = '/home/orin1/Documents/Luke/ex4/battery_n_tissue_n.csv'

data_1 = pd.read_csv(file_1)
data_2 = pd.read_csv(file_2)
data_3 = pd.read_csv(file_3)
data_4 = pd.read_csv(file_4)

data = pd.concat([data_1, data_2, data_3, data_4], axis=0)
X_object_1_new = data[['angle1', 'distance1']]
X_object_2_new = data[['angle2', 'distance2']]

scaler_1 = StandardScaler()
X_object_1_new_scaled = scaler_1.fit_transform(X_object_1_new)

scaler_2 = StandardScaler()
X_object_2_new_scaled = scaler_2.fit_transform(X_object_2_new)

y_true_object_1 = data['object1']
y_true_object_2 = data['object2']
start_time_obj_1 = time.time()
y_pred_object_1 = model_object_1.predict(X_object_1_new_scaled)
end_time_obj_1 = time.time()
latency_object_1 = end_time_obj_1 - start_time_obj_1
start_time_obj_2 = time.time()
y_pred_object_2 = model_object_2.predict(X_object_2_new_scaled)
end_time_obj_2 = time.time()
latency_object_2 = end_time_obj_2 - start_time_obj_2
y_true_combined = pd.concat([y_true_object_1, y_true_object_2], axis=0)
y_pred_combined = pd.concat([pd.Series(y_pred_object_1), pd.Series(y_pred_object_2)], axis=0)
combined_accuracy = accuracy_score(y_true_combined, y_pred_combined)
combined_f1_score = f1_score(y_true_combined, y_pred_combined, average='weighted')
combined_latency = (latency_object_1 + latency_object_2) / 2
output_file = '/home/orin1/Documents/Luke/ex4/ex4_final_new_approach/result/inference/InferenceV1-Orion.txt'
with open(output_file, 'w') as f:
    f.write(f"Combined Accuracy: {combined_accuracy * 100:.2f}%\n")
    f.write(f"Combined F1 Score: {combined_f1_score:.4f}\n")
    f.write(f"Average Inference Latency: {combined_latency:.4f} seconds\n")
print("Evaluation results saved to InferenceV5.txt")
