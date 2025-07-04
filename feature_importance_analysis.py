import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Load data
df_demo = pd.read_csv('data/processed/df_norm_demo.csv')
df_diet = pd.read_csv('data/processed/df_norm_diet.csv')
df_exam = pd.read_csv('data/processed/df_norm_exam.csv')
df_lab = pd.read_csv('data/processed/df_norm_lab.csv')
df_ques = pd.read_csv('data/processed/df_norm_ques.csv')

X_combined = np.load('data/processed/X_combined.npy')
joint_autoencoder = load_model('models/joint_autoencoder.h5', compile=False)

# Get weights from encoder layers
first_layer_weights = joint_autoencoder.layers[1].get_weights()[0]
first_layer_biases = joint_autoencoder.layers[1].get_weights()[1]
second_layer_weights = joint_autoencoder.layers[2].get_weights()[0]
second_layer_biases = joint_autoencoder.layers[2].get_weights()[1]

# Save weights
np.savetxt("results/Dense_layer_weights.csv", first_layer_weights, delimiter=",")
np.savetxt("results/Dense_layer_biases.csv", first_layer_biases, delimiter=",")
np.savetxt("results/Embedding_layer_weights.csv", second_layer_weights, delimiter=",")
np.savetxt("results/Embedding_layer_biases.csv", second_layer_biases, delimiter=",")

# Create full feature DataFrame
all_features = pd.concat([df_demo, df_diet, df_exam, df_lab, df_ques], axis=1)
feature_names = all_features.columns

# Calculate encoder-weight-based contributions
encoder_weights = pd.read_csv("results/Dense_layer_weights.csv", header=None)
latent_dim = encoder_weights.shape[1]

z_train = np.dot(all_features.values, encoder_weights.values)
gene_contributions = np.dot(z_train, encoder_weights.values.T)
gene_contributions_df = pd.DataFrame(gene_contributions, columns=feature_names)

cumulative_contribution = gene_contributions_df.abs().mean(axis=0)
cumulative_contribution_sorted = cumulative_contribution.sort_values(ascending=False)
cumulative_contribution_sorted.to_csv("results/FeatureContribution_EncoderWeight.csv")

# Plot top 20 features
plt.figure(figsize=(10, 8))
cumulative_contribution_sorted.head(20)[::-1].plot(kind='barh')
plt.title("Top 20 Features by Encoder Weight Contribution")
plt.tight_layout()
plt.savefig("results/EncoderWeight_Top20.png")
plt.show()

print("✅ Encoder weight contributions analyzed and plots saved.")

# --- Permutation Feature Importance (PFI) ---

domain_splits = [
    (0, df_demo.shape[1]),
    (df_demo.shape[1], df_demo.shape[1] + df_diet.shape[1]),
    (df_demo.shape[1] + df_diet.shape[1], df_demo.shape[1] + df_diet.shape[1] + df_exam.shape[1]),
    (df_demo.shape[1] + df_diet.shape[1] + df_exam.shape[1], df_demo.shape[1] + df_diet.shape[1] + df_exam.shape[1] + df_lab.shape[1]),
    (df_demo.shape[1] + df_diet.shape[1] + df_exam.shape[1] + df_lab.shape[1], X_combined.shape[1])
]

# Baseline error
recon = joint_autoencoder.predict(X_combined)
baseline_error = sum(mean_squared_error(X_combined[:, s:e], recon[i]) for i, (s, e) in enumerate(domain_splits))

importances = []
for i in tqdm(range(X_combined.shape[1]), desc="Calculating PFI"):
    X_perm = X_combined.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    recon_perm = joint_autoencoder.predict(X_perm)
    perm_error = sum(mean_squared_error(X_combined[:, s:e], recon_perm[j]) for j, (s, e) in enumerate(domain_splits))
    importance = perm_error - baseline_error
    importances.append(importance)

pfi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
pfi_df.sort_values("Importance", ascending=False).to_csv("results/PermutationFeatureImportance.csv", index=False)

# Plot PFI top 20
top_pfi = pfi_df.sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(top_pfi["Feature"][::-1], top_pfi["Importance"][::-1])
plt.title("Top 20 Features (PFI - Joint Autoencoder)")
plt.xlabel("Increase in Reconstruction Error (MSE)")
plt.tight_layout()
plt.savefig("results/PFI_Top20.png")
plt.show()

print("✅ Permutation Feature Importance calculated and plots saved.")
