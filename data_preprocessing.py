import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Read all modalities
df_demo = pd.read_csv('data/raw/Demo_numeric_NHANES_2005_2018.csv')
df_diet = pd.read_csv('data/raw/Diet_numeric_NHANES_2005_2018.csv')
df_exam = pd.read_csv('data/raw/Exam_numeric_NHANES_2005_2018.csv')
df_lab = pd.read_csv('data/raw/Lab_numeric_NHANES_2005_2018.csv')
df_ques = pd.read_csv('data/raw/Ques_numeric_NHANES_2005_2018.csv')

# Drop ID columns
dfs = [df_demo, df_diet, df_exam, df_lab, df_ques]
dfs_wo_id = [df.drop(columns=['SEQN']) for df in dfs]

# Fill missing values with column means
for df in dfs_wo_id:
    df.fillna(df.mean(), inplace=True)

# Normalize
scaler = MinMaxScaler()
X_demo = scaler.fit_transform(dfs_wo_id[0].values)
X_diet = scaler.fit_transform(dfs_wo_id[1].values)
X_exam = scaler.fit_transform(dfs_wo_id[2].values)
X_lab  = scaler.fit_transform(dfs_wo_id[3].values)
X_ques = scaler.fit_transform(dfs_wo_id[4].values)

# Remove column with high mean (as in your code)
df_ques_without_id = dfs_wo_id[4]
df_ques_without_id = df_ques_without_id.drop('HSAQUEX', axis=1)
X_ques = np.delete(X_ques, 7, axis=1)

# Save normalized DataFrames
pd.DataFrame(X_demo, columns=dfs_wo_id[0].columns).to_csv('data/processed/df_norm_demo.csv', index=False)
pd.DataFrame(X_diet, columns=dfs_wo_id[1].columns).to_csv('data/processed/df_norm_diet.csv', index=False)
pd.DataFrame(X_exam, columns=dfs_wo_id[2].columns).to_csv('data/processed/df_norm_exam.csv', index=False)
pd.DataFrame(X_lab, columns=dfs_wo_id[3].columns).to_csv('data/processed/df_norm_lab.csv', index=False)
pd.DataFrame(X_ques, columns=df_ques_without_id.columns).to_csv('data/processed/df_norm_ques.csv', index=False)

# Combined array for shared encoder
X_combined = np.concatenate((X_demo, X_diet, X_exam, X_lab, X_ques), axis=1)
np.save('data/processed/X_combined.npy', X_combined)

print("✅ Preprocessing done and files saved.")
