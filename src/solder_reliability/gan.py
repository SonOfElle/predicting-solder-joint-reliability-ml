"""TensorFlow GAN for synthesising solder joint reliability samples.

Faithful to the MSc thesis notebook, sections 3 and 4:

* Generator:     Dense(64) -> LeakyReLU -> Dense(128) -> LeakyReLU -> Dense(25)
* Discriminator: Dense(128) -> LeakyReLU -> Dense(64) -> LeakyReLU -> Dense(1, sigmoid)
* Adam(1e-4) for both, BinaryCrossentropy(from_logits=False)
* latent_dim=100, batch_size=32, epochs=100, num_samples=5000
* Seeds: tf=42, numpy=42, python random=42

Two deviations from the thesis notebook, both documented:

1. The original training loop only ever generated synthetic samples inside
   the ``epoch % 10 == 0`` branch, so the array returned to the caller
   depended on that branch. Here the final generator is used once after
   training to draw ``num_samples``. Same architecture, same seeds, but the
   returned array is now deterministic regardless of ``epochs``.

2. The original export reshaped 25 GAN outputs into a (7, 4) matrix in
   C-order with NaN padding to 28 cells, which is the same class of bug as
   the real-data feature layout. The export function here exposes
   ``layout="original"`` for the thesis reshape and ``layout="fixed"`` for
   the correct one. The in-memory ``(num_samples, 25)`` array is
   layout-independent and is what the sklearn models consume.

Exact numerical reproducibility of TF training is not guaranteed across
TF versions or CPU/GPU, even with seeds. The seeds fix the initialisation
and the noise draws for a given TF build. The R2 and RMSE deltas between
stages, not the absolute values, are the point of the write-up.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

SEED = 42
N_FEATURES = 25
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 100
NUM_SAMPLES = 5000
SCALER_RANGE = (0.2, 0.8)


def _set_seeds(seed: int = SEED) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_generator(
    latent_dim: int = LATENT_DIM, n_features: int = N_FEATURES
) -> tf.keras.Model:
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Dense(64, input_shape=(latent_dim,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(n_features))
    return model


def build_discriminator(n_features: int = N_FEATURES) -> tf.keras.Model:
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.Dense(128, input_shape=(n_features,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def _make_train_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_opt: tf.keras.optimizers.Optimizer,
    disc_opt: tf.keras.optimizers.Optimizer,
    latent_dim: int,
    batch_size: int,
):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def train_step(real_data):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = generator(noise, training=True)

            real_out = discriminator(real_data, training=True)
            fake_out = discriminator(generated, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_out), fake_out)
            real_loss = cross_entropy(tf.ones_like(real_out), real_out)
            fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
            disc_loss = real_loss + fake_loss

        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_opt.apply_gradients(
            zip(disc_grads, discriminator.trainable_variables)
        )
        return gen_loss, disc_loss

    return train_step


def train_gan(
    data: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    latent_dim: int = LATENT_DIM,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[tf.keras.Model, MinMaxScaler, dict]:
    """Train the GAN on ``data`` of shape ``(n_samples, 25)``.

    Returns ``(generator, scaler, history)``. ``history`` holds per-epoch
    generator and discriminator loss. The scaler is fit on the full array
    passed in, because a GAN learns a distribution, not a train split.
    Model-side scaling is handled separately by the caller.
    """
    _set_seeds(seed)

    scaler = MinMaxScaler(feature_range=SCALER_RANGE)
    X = scaler.fit_transform(data).astype(np.float32)

    n_features = X.shape[1]
    generator = build_generator(latent_dim, n_features)
    discriminator = build_discriminator(n_features)
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    train_step = _make_train_step(
        generator, discriminator, gen_opt, disc_opt, latent_dim, batch_size
    )

    history: dict[str, list[float]] = {"gen_loss": [], "disc_loss": []}
    n_batches = len(X) // batch_size

    for epoch in range(epochs):
        epoch_gen = 0.0
        epoch_disc = 0.0
        for b in range(n_batches):
            batch = X[b * batch_size : (b + 1) * batch_size]
            gen_loss, disc_loss = train_step(batch)
            epoch_gen += float(gen_loss)
            epoch_disc += float(disc_loss)

        history["gen_loss"].append(epoch_gen / n_batches)
        history["disc_loss"].append(epoch_disc / n_batches)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:3d}/{epochs}  "
                f"gen_loss={history['gen_loss'][-1]:.4f}  "
                f"disc_loss={history['disc_loss'][-1]:.4f}"
            )

    return generator, scaler, history


def generate_samples(
    generator: tf.keras.Model,
    scaler: MinMaxScaler,
    num_samples: int = NUM_SAMPLES,
    latent_dim: int = LATENT_DIM,
    seed: int = SEED,
) -> np.ndarray:
    """Draw ``num_samples`` synthetic rows in the original feature space."""
    _set_seeds(seed)
    noise = tf.random.normal([num_samples, latent_dim])
    scaled = generator(noise, training=False).numpy()
    return scaler.inverse_transform(scaled)


def export_synthetic_txt(
    synth: np.ndarray,
    out_dir: Path | str,
    layout: str = "fixed",
    fmt: str = "%.2f",
    delimiter: str = "\t",
) -> int:
    """Write ``synth`` as (7, 4) .txt files, one file per row.

    ``layout="fixed"``
        Features 0..23 into rows 0-5 in C-order, target at (6, 0), NaN in
        (6, 1:4). Matches the real data files.

    ``layout="original"``
        C-order reshape of the 25 values into (7, 4) with NaN padding to
        28 cells. Reproduces the thesis export.

    Returns the number of files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(synth.shape[0])

    if layout == "fixed":
        for i in range(n):
            row = synth[i]
            matrix = np.full((7, 4), np.nan)
            matrix[:6, :] = row[:24].reshape(6, 4)
            matrix[6, 0] = row[24]
            np.savetxt(
                out_dir / f"Data_Matrix_{i + 1}.txt",
                matrix,
                fmt=fmt,
                delimiter=delimiter,
            )
    elif layout == "original":
        for i in range(n):
            padded = np.full(28, np.nan)
            padded[:25] = synth[i]
            matrix = padded.reshape(7, 4)
            np.savetxt(
                out_dir / f"Data_Matrix_{i + 1}.txt",
                matrix,
                fmt=fmt,
                delimiter=delimiter,
            )
    else:
        raise ValueError(
            f"layout must be 'fixed' or 'original', got {layout!r}"
        )

    return n


def run(
    data: np.ndarray,
    out_dir: Path | str | None = None,
    layout: str = "fixed",
    epochs: int = EPOCHS,
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Train, generate, optionally export.

    Returns ``(synth, history)``. ``synth`` is ``(num_samples, 25)`` in the
    original feature space and is independent of ``layout``.
    """
    generator, scaler, history = train_gan(
        data, epochs=epochs, seed=seed, verbose=verbose
    )
    synth = generate_samples(
        generator, scaler, num_samples=num_samples, seed=seed
    )
    if out_dir is not None:
        n = export_synthetic_txt(synth, out_dir, layout=layout)
        if verbose:
            print(f"Wrote {n} files to {out_dir} (layout={layout})")
    return synth, history