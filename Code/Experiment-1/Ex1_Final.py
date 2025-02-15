import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

df_list = [
    pd.read_csv('recycle_ex1_n.csv'),
    pd.read_csv('recycle_ex1_s.csv'),
    pd.read_csv('nvidia_ex1_n.csv'),
    pd.read_csv('nvidia_ex1_s.csv'),
    pd.read_csv('battery_ex1_n.csv'),
    pd.read_csv('battery_ex1_s.csv'),
    pd.read_csv('tissue_ex1_n.csv'),
    pd.read_csv('tissue_ex1_s.csv')
]
df = pd.concat(df_list, ignore_index=True)

print(df['object'].isnull().sum())

df = df.dropna(subset=['object'])

X = df[['angle', 'distance (mm)']]
y = df['object']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)
df2 = pd.concat([
    pd.read_csv('battery_ex1_n_inference.csv'),
    pd.read_csv('battery_ex1_s_inference.csv')
], ignore_index=True)
df2 = df2.dropna(subset=['object'])
X2 = df2[['angle', 'distance (mm)']]
y2 = df2['object']
gamma_values = [100]
results_file = 'svm_ex1_gamma_results.csv'

with open(results_file, 'w') as file:
    file.write('gamma,train_accuracy,test_accuracy,inference_accuracy,train_f1,test_f1,inference_f1,inference_latency\n')

    for gamma in gamma_values:
        svm_model = SVC(kernel='rbf', gamma=gamma)
        svm_model.fit(X_train, y_train)
        model_filename = f'svm_model_ex1_gamma_{gamma}.pkl'
        joblib.dump(svm_model, model_filename)
        y_train_pred = svm_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        y_test_pred = svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        loaded_model = joblib.load(model_filename)
        start_time = time.time()
        y2_pred = loaded_model.predict(X2)
        end_time = time.time()
        inference_latency = end_time - start_time
        inference_accuracy = accuracy_score(y2, y2_pred)
        inference_f1 = f1_score(y2, y2_pred, average='weighted')
        print(f'Gamma: {gamma}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Inference Accuracy: {inference_accuracy:.4f}')
        print(f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Inference F1: {inference_f1:.4f}')
        print(f'Inference Latency: {inference_latency:.4f} seconds')
        print('Classification Report for Test Set:\n', classification_report(y_test, y_test_pred))
        print('Classification Report for Inference Data:\n', classification_report(y2, y2_pred))
        file.write(f'{gamma},{train_accuracy},{test_accuracy},{inference_accuracy},{train_f1},{test_f1},{inference_f1},{inference_latency}\n')
print(f'Results saved to {results_file}')