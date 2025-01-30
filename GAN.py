import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv('C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\Projects\\Drug Effect Simulation\\medicine_dataset.csv')

import pandas as pd
from sklearn.preprocessing import LabelEncoder

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')
# Assuming columns: 'Chemical Class', 'Therapeutic Class', 'Action Class' are categorical
categorical_columns = ['ChemicalClass', 'TherapeuticClass', 'ActionClass']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    # Fill missing values with a placeholder (e.g., 'Unknown') before label encoding
    df[col].fillna('Unknown', inplace=True)
    df[col] = label_encoder.fit_transform(df[col])

# Now the categorical columns are encoded with integer labels
# You can use df.head() to verify the changes
print(df.head())
