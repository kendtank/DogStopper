# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 16:09
@Author  : Kend
@FileName: model_ge2e
@Software: PyCharm
@modifier:
"""



# 负责：embedding backbone (raw 32D) + GE2E layer/loss + cosine utils


import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ----- constants (keep consistent with dataset_utils) -----
INPUT_TIME = 18    # time frames (200ms with 25/10ms -> ~18 frames)
N_MELS = 40
EMBED_DIM = 32

# ---------------- backbone (raw embedding, no L2, no BN) ----------------
def build_backbone_raw_embedding(input_time=INPUT_TIME, n_mels=N_MELS, embed_dim=EMBED_DIM):
    inp = layers.Input(shape=(input_time, n_mels, 1), name="logmel_input")
    x = layers.Conv2D(8, (3,3), padding='same', activation='relu')(inp)
    x = layers.SeparableConv2D(8, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(16, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.AveragePooling2D((4,10))(x)   # reduce spatial dims
    x = layers.Reshape((16,))(x)
    emb = layers.Dense(embed_dim, name="embedding_output")(x)  # raw embedding
    model = Model(inp, emb, name="backbone_raw")
    return model

# ---------------- GE2E Layer (computes scaled similarities) ----------------
class GE2ELayer(tf.keras.layers.Layer):
    """
    Input: embeddings shape (N, M, D)  - NOT flattened
    Output: logits shape (N*M, N) for classification (softmax over speakers)
    It stores trainable scale w and bias b (initialized per paper)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initial values like in paper
        self.init_w = 10.0
        self.init_b = -5.0

    def build(self, input_shape):
        # input_shape = (N, M, D)
        D = int(input_shape[-1])
        self.w = self.add_weight(name="w", shape=(), initializer=tf.keras.initializers.Constant(self.init_w), trainable=True)
        self.b = self.add_weight(name="b", shape=(), initializer=tf.keras.initializers.Constant(self.init_b), trainable=True)
        super().build(input_shape)

    def call(self, embeddings):
        """
        embeddings: shape (N, M, D)
        returns: logits (N*M, N)
        """
        # L2 normalize embeddings along D
        emb_norm = tf.nn.l2_normalize(embeddings, axis=-1)  # (N,M,D)

        # centroids for each speaker across M (mean)
        centroids = tf.reduce_mean(emb_norm, axis=1, keepdims=False)  # (N, D)

        N = tf.shape(emb_norm)[0]
        M = tf.shape(emb_norm)[1]
        D = tf.shape(emb_norm)[2]

        # For each e_ij compute c_i_minus = (sum_m e_im - e_ij)/(M-1)
        # Expand to (N,M,D)
        emb_sum = tf.reduce_sum(emb_norm, axis=1, keepdims=True)  # (N,1,D)
        c_i_minus = (emb_sum - emb_norm) / tf.cast((tf.cast(M, tf.float32) - 1.0), tf.float32)  # (N,M,D)

        # prepare centroids for cross similarity:
        # centroids for other speakers: shape (1,1,N,D) after expand
        centroids_exp = tf.reshape(centroids, (1,1, -1, D))  # (1,1,N,D)

        # build c_k for every e_ij:
        # For own speaker, use c_i_minus; for others use centroids
        # construct c_k_for_each e_ij with shape (N,M,N,D)
        emb_norm_exp = tf.expand_dims(emb_norm, axis=2)  # (N,M,1,D)
        c_i_minus_exp = tf.expand_dims(c_i_minus, axis=2)  # (N,M,1,D)

        # base centroids tiled to (N,M,N,D)
        centroids_tiled = tf.tile(centroids_exp, [N, M, 1, 1])  # (N,M,N,D)

        # replace diagonal positions with c_i_minus
        # create mask of shape (N,N) with ones on diagonal -> expand to (N,M,N,1)
        eye = tf.eye(N)
        eye_exp = tf.reshape(eye, (N,1,N,1))  # (N,1,N,1)
        eye_exp = tf.tile(eye_exp, [1,M,1,1])  # (N,M,N,1)
        c_ks = tf.where(tf.cast(eye_exp, tf.bool), c_i_minus_exp, centroids_tiled)  # (N,M,N,D)

        # now compute cosine similarity between emb_norm_exp (N,M,1,D) and c_ks (N,M,N,D) along D
        emb_norm_for_dot = emb_norm_exp  # (N,M,1,D)
        # dot product:
        dot = tf.reduce_sum(emb_norm_for_dot * c_ks, axis=-1)  # (N,M,N) -> cosines
        # scale + bias
        logits = self.w * dot + self.b  # (N,M,N)
        # flatten to (N*M, N)
        logits_flat = tf.reshape(logits, (N * M, N))
        return logits_flat

# ---------------- convenience function to flatten batch (N,M)->(N*M,...) and labels ----------------
def flatten_logits_and_labels(logits_nm_n, N, M):
    # logits_nm_n: (N*M, N) already
    # labels: for training each e_ij has true label i (0..N-1)
    # labels = tf.repeat(tf.range(N), repeats=M)
    labels = tf.repeat(tf.range(N, dtype=tf.int64), repeats=M)

    return logits_nm_n, labels

# ---------------- cosine similarity util for inference ----------------
def cosine_similarity(a, b):
    # a: (D,) normalized or not; b: (K, D) or (D,)
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12) if b.ndim==2 else b/(np.linalg.norm(b)+1e-12)
    return np.dot(b_n, a_n)  # if b is (K,D) -> returns (K,)
