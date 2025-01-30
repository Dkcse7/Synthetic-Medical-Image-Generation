import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Drug Effect Simulation\\medicine_dataset.csv', low_memory=False)

# Label encode categorical columns
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Define input and output dimensions
input_dim = len(df.columns)  # Number of features
output_dim = input_dim

# Generator model
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

# Discriminator model
def build_discriminator(input_dim):
    model = tf.keras.Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile discriminator model
discriminator = build_discriminator(input_dim)
discriminator.compile(loss='hinge', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002), metrics=['accuracy'])

# GAN architecture
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(input_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002))
    return gan

# Compile GAN
generator = build_generator(input_dim, output_dim)
gan = build_gan(generator, discriminator)

# Training parameters
epochs = 10000
batch_size = 32

# Training loop
for epoch in range(epochs):
    # Generate random noise as input for the generator
    noise = np.random.normal(0, 1, (batch_size, input_dim))

    # Generate synthetic drug data
    synthetic_data = generator.predict(noise)

    # Train discriminator on real data
    idx = np.random.randint(0, len(df), batch_size)
    real_data = df.iloc[idx]
    real_labels = np.ones((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)

    # Train discriminator on synthetic data
    fake_labels = np.zeros((batch_size, 1))
    d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)

    # Aggregate discriminator loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator (via GAN)
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress and evaluation metrics
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss[0]}, Discriminator Accuracy: {d_loss[1]}, Generator Loss: {g_loss}")


#Save Model
generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')