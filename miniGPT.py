import tensorflow as tf
import numpy as np
import os

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ------------------------------
# Data Preparation
# ------------------------------
with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# ------------------------------
# Hyperparameters
# ------------------------------
vocab = sorted(set(text))
vocab_size = len(vocab)
embed_dim = 256
num_heads = 4
ff_dim = 128
num_layers = 4
block_size = 128
batch_size = 16
dropout_rate = 0.1
learning_rate = 1e-4

# ------------------------------
# Tokenization
# ------------------------------
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}

def encode(s):
    return [char2idx[c] for c in s]

def decode(ids):
    return ''.join([idx2char[i] for i in ids])



data = np.array(encode(text), dtype=np.int32)
print("Dataset size:", len(data))

def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

# ------------------------------
# Positional Embedding
# ------------------------------
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(1024, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions

# ------------------------------
# Causal Self-Attention Block
# ------------------------------
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                       key_dim=embed_dim // num_heads,
                                                       dropout=dropout_rate)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return self.attn(x, x, attention_mask=causal_mask)

# ------------------------------
# Transformer Block
# ------------------------------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ------------------------------
# GPT Model
# ------------------------------
class MiniGPT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = PositionalEmbedding(vocab_size, embed_dim, block_size)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
                                   for _ in range(num_layers)]
        self.norm = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_layer(x)

# ------------------------------
# Training
# ------------------------------
model = MiniGPT()
dummy_input = tf.zeros((1, block_size), dtype=tf.int32)
model(dummy_input) 

# Load weights if available
if os.path.exists("minigpt.weights.h5"):
    model.load_weights("minigpt.weights.h5")
    print("✅ Loaded saved weights.")
else:
    print("ℹ️ No saved weights found. Starting from scratch.")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

print("Training...")
for step in range(5000):
    x_batch, y_batch = get_batch(data, block_size, batch_size)
    loss = train_step(x_batch, y_batch)
    if step % 100 == 0: 
        print(f"{step/50}%, Loss: {loss:.4f}")

model.save_weights("minigpt.weights.h5")


# ------------------------------
# Generation
# ------------------------------
def generate(model, prompt, max_new_tokens=100):
    #model.reset_states()
    input_ids = tf.constant([encode(prompt)], dtype=tf.int32)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = tf.cast(tf.random.categorical(next_token_logits, num_samples=1), tf.int32)
        input_ids = tf.concat([input_ids, next_token], axis=-1)
    return decode(input_ids.numpy().squeeze())

print("\nGenerated Text:")
print(generate(model, "hello ", max_new_tokens=100))
