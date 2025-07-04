import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

X_combined = np.load('data/processed/X_combined.npy')

input_dim = X_combined.shape[1]
latent_dim = 3
hidden_dim = 64

encoder_input = Input(shape=(input_dim,))
h = Dense(hidden_dim, activation='relu')(encoder_input)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

def build_decoder(out_dim, latent_dim):
    d_input = Input(shape=(latent_dim,))
    h = Dense(hidden_dim, activation='relu')(d_input)
    out = Dense(out_dim, activation='sigmoid')(h)
    return Model(d_input, out)

# Define sub-dimensions
dims = [pd.read_csv(f"data/processed/df_norm_{mod}.csv").shape[1]
        for mod in ['demo', 'diet', 'exam', 'lab', 'ques']]

decoders = [build_decoder(dim, latent_dim) for dim in dims]

outputs = [dec(z) for dec in decoders]
joint_autoencoder = Model(encoder_input, outputs)
joint_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')

split_idxs = np.cumsum(dims)
slices = np.split(X_combined, split_idxs[:-1], axis=1)

joint_autoencoder.fit(X_combined, slices, epochs=100, batch_size=16, verbose=2)

encoder = Model(encoder_input, z)
encoded_combined = encoder.predict(X_combined)
np.save('results/encoded_multidecoder.npy', encoded_combined)

print("✅ Joint multi-decoder AE trained and embeddings saved.")
