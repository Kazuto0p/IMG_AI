import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Accessing Keras components through TensorFlow
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers


import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os

# Constants
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10000
SAMPLE_INTERVAL = 1000
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999
DROPOUT_RATE = 0.3
LAMBDA = 10  # Gradient penalty lambda hyperparameter

# Create a directory for saving generated images
if not os.path.exists("images"):
    os.makedirs("images")

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

# Generator model with residual blocks
def residual_block(x, filters):
    res = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    res = tfa.layers.InstanceNormalization()(res)
    res = layers.LeakyReLU()(res)
    res = layers.Conv2D(filters, kernel_size=3, padding='same')(res)
    res = tfa.layers.InstanceNormalization()(res)
    return layers.Add()([x, res])

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    
    for _ in range(2):
        model.add(layers.Lambda(lambda x: residual_block(x, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Discriminator model with Spectral Normalization, Gaussian Noise, and Gradient Penalty
def build_discriminator():
    model = models.Sequential()
    model.add(layers.GaussianNoise(0.1, input_shape=[28, 28, 1]))
    model.add(tfa.layers.SpectralNormalization(
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(tfa.layers.SpectralNormalization(
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    ))
    model.add(tfa.layers.InstanceNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# WGAN loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Gradient penalty implementation
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = real_images.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    return tf.reduce_mean((grad_norm - 1.0) ** 2)

# Function to save and display generated images
def sample_images(generator, epoch, image_grid_rows=4, image_grid_columns=4):
    noise = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, LATENT_DIM))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale from [-1, 1] to [0, 1]

    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharex=True, sharey=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    plt.savefig(f"images/epoch_{epoch}.png")  # Save the plot
    plt.close()  # Close the figure

# Load and preprocess the data
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train, axis=-1)

# Apply data augmentation
X_train = data_augmentation(X_train)

# Convert data to Dataset API
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Build and compile the discriminator
discriminator = build_discriminator()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.9
)
d_optimizer = optimizers.AdamW(learning_rate=lr_schedule, beta_1=BETA_1, beta_2=BETA_2)
discriminator.compile(loss=wasserstein_loss, optimizer=d_optimizer)

# Build the generator
generator = build_generator()

# Keep the discriminator untrainable in the GAN context
discriminator.trainable = False

# Build and compile the GAN
gan = models.Sequential([generator, discriminator])
g_optimizer = optimizers.AdamW(learning_rate=lr_schedule, beta_1=BETA_1, beta_2=BETA_2)
gan.compile(loss=wasserstein_loss, optimizer=g_optimizer)

# Callbacks for training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/gan_checkpoint.h5",
    save_best_only=True,
    monitor="g_loss",
    verbose=1
)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='g_loss', factor=0.5, patience=10, verbose=1
)

log_dir = "logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Training function
def train_gan(generator, discriminator, gan, dataset, epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL):
    for epoch in range(epochs):
        for real_imgs in dataset:
            # Train Discriminator
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
            fake_imgs = generator.predict(noise)

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_imgs, -np.ones((BATCH_SIZE, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_imgs, np.ones((BATCH_SIZE, 1)))

            gp = gradient_penalty(discriminator, real_imgs, fake_imgs)
            d_loss = d_loss_fake + LAMBDA * gp - d_loss_real

            # Train Generator
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, -np.ones((BATCH_SIZE, 1)))

        # Print progress and save images at intervals
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
            sample_images(generator, epoch)

        # Run callbacks
        lr_callback.on_epoch_end(epoch, logs={"g_loss": g_loss})
        checkpoint_callback.on_epoch_end(epoch, logs={"g_loss": g_loss})
        tensorboard_callback.on_epoch_end(epoch, logs={"g_loss": g_loss})

# Train the GAN
train_gan(generator, discriminator, gan, dataset)
