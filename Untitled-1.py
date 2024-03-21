import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import pymongo
from pymongo import MongoClient

# base64_encoded_data= ""
# # Decode Base64 Image Data
# image_data = base64.b64decode(base64_encoded_data)
image = Image.open("captured_image.jpg")

# Preprocess Image
image = image.resize((299, 299))  # InceptionV3 input size
image_array = np.array(image) / 255.0  # Normalize pixel values

# Use Pre-trained Model (InceptionV3)
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Expand dimensions to match model input shape
input_data = np.expand_dims(image_array, axis=0)

# Extract Embeddings
embeddings = model.predict(input_data)

print("Embeddings shape:", embeddings.shape)
print(embeddings)


known_encoding_list = embeddings.tolist()




# Create a connection
client = MongoClient('localhost', 27017)

# Accessing a database
db = client['major']

# Accessing a collection
collection = db['embedd']

document = {
    'embedding':known_encoding_list
    
}
result = collection.insert_one(document)

# Print the unique identifier for the inserted document
print(result.inserted_id)

# Close the connection
client.close()
