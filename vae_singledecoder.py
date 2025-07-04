import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

X_combined = np.load('data/processed/X_combined.npy')
input_dim = X_combined.shape[1]
latent_dim = 3
hidden_dim = 64

inputs = Input(shape=(input_dim,))
h = Dense(hidden_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])
decoder_h = Dense(hidden_dim, activation='relu')
decoder_out = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

vae = Model(inputs, outputs)

def vae_loss(x, x_decoded):
    recon = binary_crossentropy(x, x_decoded) * input_dim
    kl = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(recon + kl)

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(X_combined, X_combined, epochs=30, batch_size=32)

encoder = Model(inputs, z)
encoded_combined = encoder.predict(X_combined)
np.save('results/encoded_singledecoder.npy', encoded_combined)

print("✅ Single-decoder VAE trained and embeddings saved.")
