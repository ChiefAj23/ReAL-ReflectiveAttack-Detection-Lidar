import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# 4 boxes:
# recycle
# nvidia
# battery
# tissue
#
# all combos:
# rn, rb, rt
# nr, nb, nt
# br, bn, bt
# tr, tn, tb
# Each combo has another 4 combinations for N/S
# 48 tests
#
# Alternatively:
# Only train on recycle and nvidia combos
# Inference on battery and tissue combos
# all combos:
# rn, nr
# bt, tb
# Each combo has another 4 combinations for N/S
# 16 tests
df_list = []
df_list.append(pd.read_csv('recycle_n_nvidia_n.csv'))
df_list.append(pd.read_csv('recycle_n_nvidia_s.csv'))
df_list.append(pd.read_csv('recycle_s_nvidia_n.csv'))
df_list.append(pd.read_csv('recycle_s_nvidia_s.csv'))

df_list.append(pd.read_csv('nvidia_n_recycle_n.csv'))
df_list.append(pd.read_csv('nvidia_n_recycle_s.csv'))
df_list.append(pd.read_csv('nvidia_s_recycle_n.csv'))
df_list.append(pd.read_csv('nvidia_s_recycle_s.csv'))

df = pd.concat(df_list, ignore_index=True)
assert df['object1'].isnull().sum() == 0, f"{df['object1'].isnull().sum()} rows invalid"
assert df['object2'].isnull().sum() == 0, f"{df['object2'].isnull().sum()} rows invalid"
df = df.dropna(subset=['object1'])
df = df.dropna(subset=['object2'])
X = df[['angle1', 'distance1', 'angle2', 'distance2']]
y = df['object1', 'object2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
df2_list = []
df2_list.append(pd.read_csv('battery_n_tissue_n.csv'))
df2_list.append(pd.read_csv('battery_n_tissue_s.csv'))
df2_list.append(pd.read_csv('battery_s_tissue_n.csv'))
df2_list.append(pd.read_csv('battery_s_tissue_s.csv'))

df2_list.append(pd.read_csv('tissue_n_battery_n.csv'))
df2_list.append(pd.read_csv('tissue_n_battery_s.csv'))
df2_list.append(pd.read_csv('tissue_s_battery_n.csv'))
df2_list.append(pd.read_csv('tissue_s_battery_s.csv'))

df2 = pd.concat(df2_list, ignore_index=True)
assert df2['object1'].isnull().sum() == 0, f"{df2['object1'].isnull().sum()} rows invalid"
assert df2['object2'].isnull().sum() == 0, f"{df2['object2'].isnull().sum()} rows invalid"

df2 = df2.dropna(subset=['object1'])
df2 = df2.dropna(subset=['object2'])

X2 = df2[['angle1', 'distance1', 'angle2', 'distance2']]
y2 = df2['object1', 'object2']

gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100]
results_file = 'svm_gamma_results.csv'

with open(results_file, 'w') as file:
    file.write('gamma,train_accuracy,test_accuracy,inference_accuracy,train_f1,test_f1,inference_f1\n')

    for gamma in gamma_values:
        svm_model = SVC(kernel='rbf', gamma=gamma)
        svm_model.fit(X_train, y_train)
        model_filename = f'svm_model_gamma_{gamma}.pkl'
        joblib.dump(svm_model, model_filename)
        y_train_pred = svm_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        y_test_pred = svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        loaded_model = joblib.load(model_filename)
        y2_pred = loaded_model.predict(X2)
        inference_accuracy = accuracy_score(y2, y2_pred)
        inference_f1 = f1_score(y2, y2_pred, average='weighted')
        print(f'Gamma: {gamma}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Inference Accuracy: {inference_accuracy:.4f}')
        print(f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Inference F1: {inference_f1:.4f}')
        print('Classification Report for Test Set:\n', classification_report(y_test, y_test_pred))
        print('Classification Report for Inference Data:\n', classification_report(y2, y2_pred))

        file.write(f'{gamma},{train_accuracy},{test_accuracy},{inference_accuracy},{train_f1},{test_f1},{inference_f1}\n')

print(f'Results saved to {results_file}')

