import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# Load data
df_label = pd.read_csv('data/raw/Label_NHANES_2005_2018.csv')
df_clusters = pd.read_csv('results/Cluster_labels.csv')
df_demo = pd.read_csv('data/processed/df_norm_demo.csv')
df_diet = pd.read_csv('data/processed/df_norm_diet.csv')
df_exam = pd.read_csv('data/processed/df_norm_exam.csv')
df_lab = pd.read_csv('data/processed/df_norm_lab.csv')
df_ques = pd.read_csv('data/processed/df_norm_ques.csv')

X_all = pd.concat([df_demo, df_diet, df_exam, df_lab, df_ques], axis=1)
X_all["Cluster"] = df_clusters["Cluster_Labels_3"]
y_all = df_label["DEPR_RISK"]

results = []
for cluster_id in sorted(X_all["Cluster"].unique()):
    print(f"\n📦 Cluster {cluster_id} ---------------------")

    mask = X_all["Cluster"] == cluster_id
    X_cluster = X_all.loc[mask].drop(columns=["Cluster"])
    y_cluster = y_all.loc[mask]

    if len(y_cluster) < 50 or y_cluster.nunique() < 2:
        print("⚠️ Skipping due to low samples or no label variation")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.3, stratify=y_cluster, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    neg, pos = y_train_res.value_counts()
    scale_pos_weight = neg / pos

    xgb = XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=4,
                        scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                        eval_metric='logloss', random_state=42)
    xgb.fit(X_train_res, y_train_res)

    y_prob = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"🎯 ROC-AUC: {auc:.3f}")

    # PFI on training
    pfi = permutation_importance(xgb, X_train_res, y_train_res, scoring='roc_auc', n_repeats=5, random_state=42)
    pfi_df = pd.DataFrame({"Feature": X_cluster.columns, "Importance": pfi.importances_mean})
    top_feats = pfi_df.sort_values("Importance", ascending=False).head(10)
    print("🔥 Top 10 features:")
    print(top_feats)

    results.append({
        "Cluster": cluster_id,
        "ROC-AUC": auc,
        "Top_Features": top_feats["Feature"].tolist()
    })

# Save summary
summary_df = pd.DataFrame(results)
summary_df.to_csv("results/Cluster_Model_Summary.csv", index=False)

print("\n✅ Cluster-specific models trained and summary saved.")
