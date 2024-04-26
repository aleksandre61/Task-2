# data_processor2
import pandas as pd
import ipaddress
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

# reading the dataset
df = pd.read_csv("Darknet.csv")

# droping the colums no longer need
df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)

# droping rows with missing values
df = df.dropna()

# encoding the label column
label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

# keeping the preprocessed dataset
df.to_csv("processed.csv", index=False)

df = pd.read_csv("processed.csv")

# splitting features and label
features = df.drop(['Label1'], axis=1)
label = df['Label1']

# spliting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
# compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test))

# training the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# keeping it into a new file
with open('accuracy2.txt', 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')