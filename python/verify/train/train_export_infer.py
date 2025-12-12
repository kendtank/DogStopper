# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 16:10
@Author  : Kend
@FileName: train_export_infer
@Software: PyCharm
@modifier:
"""


# Run: python train_export_infer.py

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from dataset_utils import build_speaker_bank, n_m_batch_generator, batch_wave_to_logmel
from model_ge2e import build_backbone_raw_embedding, GE2ELayer, flatten_logits_and_labels, cosine_similarity

# ---------------- hyperparams ----------------
DATASET_ROOT = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog"
N = 8    # speakers per batch (adjust to your class count <= available)
M = 4    # utterances per speaker
EPOCHS = 30
STEPS_PER_EPOCH = 50
LR = 1e-3
BATCH_SAVE_DIR = "dog_ge2e_ckpt"
EXPORT_DIR = "dog_bark_embed32_export"

os.makedirs(BATCH_SAVE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- prepare data ----------------
print("Building speaker bank ...")
speaker_bank = build_speaker_bank(DATASET_ROOT)
print("Speakers with >=M segments:", [k for k,v in speaker_bank.items() if len(v)>=M])
print("Total speakers in bank:", len(speaker_bank))
gen = n_m_batch_generator(speaker_bank, N=N, M=M, infinite=True)

# ---------------- build model bits ----------------
backbone = build_backbone_raw_embedding()
ge2e = GE2ELayer()
# warm build
dummy_batch = tf.zeros((N, M, backbone.output_shape[-1]), dtype=tf.float32)
_ = ge2e(dummy_batch)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

best_val = -1.0

# ---------------- training loop ----------------
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    epoch_losses = []
    epoch_accs = []
    for step in range(1, STEPS_PER_EPOCH+1):
        batch_wave, chosen_speakers = next(gen)  # (N*M, SEG_SAMPLES)
        # convert to logmel tensor (N*M, time, n_mels,1)
        batch_logmel = batch_wave_to_logmel(batch_wave)
        B = batch_logmel.shape[0]
        # reshape to (N, M, time, n_mels,1)
        bm = batch_logmel.reshape((N, M) + batch_logmel.shape[1:])
        # compute embeddings per utterance
        # flatten to (N*M, time, n_mels,1) then pass through backbone
        flat = bm.reshape((N*M, ) + batch_logmel.shape[1:])
        emb_flat = backbone(flat, training=True)  # (N*M, D)
        emb_nm_d = tf.reshape(emb_flat, (N, M, -1))  # (N, M, D)
        # GE2E logits
        logits = ge2e( tf.nn.l2_normalize(emb_nm_d, axis=-1) * 1.0 )  # (N*M, N), but ge2e also L2s internally; here safe
        labels = tf.cast(tf.repeat(tf.range(N), repeats=M), tf.int64)
        with tf.GradientTape() as tape:
            emb_flat = backbone(flat, training=True)  # 放进 tape
            emb_nm_d = tf.reshape(emb_flat, (N, M, -1))
            logits = ge2e(emb_nm_d)
            loss = loss_fn(labels, logits)
        vars_all = backbone.trainable_variables + ge2e.trainable_variables
        grads = tape.gradient(loss, vars_all)
        optimizer.apply_gradients(zip(grads, vars_all))
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()
        epoch_losses.append(float(loss.numpy()))
        epoch_accs.append(float(acc))

    # end epoch
    avg_loss = float(np.mean(epoch_losses))
    avg_acc = float(np.mean(epoch_accs))
    print(f"Epoch {epoch}/{EPOCHS} loss={avg_loss:.4f} id_acc={avg_acc:.4f} time={time.time()-t0:.1f}s")

    # checkpoint every epoch
    backbone.save_weights(os.path.join(BATCH_SAVE_DIR, "backbone_latest.weights.h5"))
    np.save(os.path.join(BATCH_SAVE_DIR, "ge2e_w.npy"), ge2e.w.numpy())
    np.save(os.path.join(BATCH_SAVE_DIR, "ge2e_b.npy"), ge2e.b.numpy())

# ---------------- export embedding model for TFLite ----------------
print("Saving final embedding model (raw output, no L2) ...")
export_model = build_backbone_raw_embedding()
export_model.set_weights(backbone.get_weights())
# save keras native (use .keras ext)
export_keras_path = os.path.join(EXPORT_DIR, "embedding_model.keras")
export_model.save(export_keras_path)

# TFLite conversion (FP32)
print("Converting to TFLite (FP32) ...")
converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
tflite_model = converter.convert()
tflite_path = os.path.join(EXPORT_DIR, "embedding_model_fp32.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite:", tflite_path)

# ---------------- inference / verification example ----------------
def make_embedding_from_wav(wav_path):
    from dataset_utils import read_wave_mcu_style_float, sliding_windows_200ms_50ms, batch_wave_to_logmel
    w = read_wave_mcu_style_float(wav_path)
    segs = sliding_windows_200ms_50ms(w)
    # choose first segment
    seg = segs[0:1]
    arr = np.stack(seg, axis=0).astype(np.float32)
    logmel = batch_wave_to_logmel(arr)  # (1, time, n_mels,1)
    emb = export_model.predict(logmel)
    # L2 normalize for comparison on device side
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb_n[0]

# demo: build small template bank and compare
if __name__ == "__main__":
    # pick two WAVs for demo
    demo_a = list(speaker_bank.values())[0][0]  # raw waveform of first seg
    # actually save to temp and call make_embedding... simpler: convert directly
    # create embedding for two samples:
    # Here show how to compare two embeddings:
    print("Demo cosine comparison: (random vectors)")
    a = np.random.randn(32)
    b = np.random.randn(32)
    print("cosine:", cosine_similarity(a, b))
