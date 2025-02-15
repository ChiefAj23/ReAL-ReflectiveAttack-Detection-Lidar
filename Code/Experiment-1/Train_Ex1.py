import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm

model_dir = 'svm_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_files = [
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/recycle_ex1_n.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/recycle_ex1_s.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/nvidia_ex1_n.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/nvidia_ex1_s.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_n.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_s.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/tissue_ex1_n.csv',
    '/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/tissue_ex1_s.csv'
]
df_list = [pd.read_csv(file) for file in train_files]
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['object'])
X = df[['angle', 'distance (mm)']]
y = df['object']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
inference_n = pd.read_csv('/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_n_inference.csv').dropna(subset=['object'])
inference_s = pd.read_csv('/Users/chiefaj/Tntech-Research/REU/Luke/Aug18/experiments/ex1/battery_ex1_s_inference.csv').dropna(subset=['object'])
X_inference_n = scaler.transform(inference_n[['angle', 'distance (mm)']])
y_inference_n = inference_n['object']
X_inference_s = scaler.transform(inference_s[['angle', 'distance (mm)']])
y_inference_s = inference_s['object']
gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1]
C_values = [0.1, 1, 10, 100]
total_combinations = len(gamma_values) * len(C_values)
best_f1_score = 0
best_model = None
best_params = {}

with open('svm_inference_results.txt', 'w') as file:
    with tqdm(total=total_combinations, desc="Training Progress", unit="model") as pbar:
        for gamma in gamma_values:
            for C in C_values:
                svm_model = SVC(kernel='rbf', gamma=gamma, C=C)
                svm_model.fit(X_train, y_train)
                y_train_pred = svm_model.predict(X_train)
                y_test_pred = svm_model.predict(X_test)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                y_inference_n_pred = svm_model.predict(X_inference_n)
                y_inference_s_pred = svm_model.predict(X_inference_s)
                inference_n_accuracy = accuracy_score(y_inference_n, y_inference_n_pred)
                inference_s_accuracy = accuracy_score(y_inference_s, y_inference_s_pred)
                inference_n_f1 = f1_score(y_inference_n, y_inference_n_pred, average='weighted')
                inference_s_f1 = f1_score(y_inference_s, y_inference_s_pred, average='weighted')
                avg_inference_f1 = (inference_n_f1 + inference_s_f1) / 2

                if (test_f1 + avg_inference_f1) / 2 > best_f1_score:
                    best_f1_score = (test_f1 + avg_inference_f1) / 2
                    best_model = svm_model
                    best_params = {'gamma': gamma, 'C': C}
                    model_filename = os.path.join(model_dir, f'svm_model_gamma_{gamma}_C_{C}.pkl')
                    joblib.dump(svm_model, model_filename)

                file.write(f'gamma={gamma}, C={C}\n')
                file.write(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}\n')
                file.write(f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}\n')
                file.write(f'Inference Battery Normal Accuracy: {inference_n_accuracy:.4f}, F1: {inference_n_f1:.4f}\n')
                file.write(f'Inference Battery Specular Accuracy: {inference_s_accuracy:.4f}, F1: {inference_s_f1:.4f}\n')
                file.write('\nClassification Report for Test Set:\n')
                file.write(classification_report(y_test, y_test_pred))
                file.write('\nClassification Report for Battery Normal Inference Data:\n')
                file.write(classification_report(y_inference_n, y_inference_n_pred))
                file.write('\nClassification Report for Battery Specular Inference Data:\n')
                file.write(classification_report(y_inference_s, y_inference_s_pred))
                file.write("\n" + "="*50 + "\n\n")
                pbar.update(1)
    file.write(f'\nBest Model: gamma={best_params["gamma"]}, C={best_params["C"]}\n')
    file.write(f'Best Model Test and Inference Avg F1 Score: {best_f1_score:.4f}\n')
print("Training complete. Results saved to svm_inference_results.txt and best model saved in 'svm_models' directory.")