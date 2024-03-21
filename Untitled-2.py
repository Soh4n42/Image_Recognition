import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import pymongo
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity


# base64_encoded_data= ""
# # Decode Base64 Image Data
# image_data = base64.b64decode(base64_encoded_data)
image = Image.open("verification_image.jpg")

# Preprocess Image
image = image.resize((299, 299))  # InceptionV3 input size
image_array = np.array(image) / 255.0  # Normalize pixel values

# Use Pre-trained Model (InceptionV3)
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Expand dimensions to match model input shape
input_data = np.expand_dims(image_array, axis=0)

# Extract Embeddings
embeddings1 = model.predict(input_data)

print("Embeddings shape:", embeddings1.shape)
print(embeddings1)

# Create a connection
client = MongoClient('localhost', 27017)

# Accessing a database
db = client['major']

# Accessing a collection
collection = db['embedd']



# Retrieve the stored embeddings from the database
stored_embeddings = []
for document in collection.find():
    embedding = np.array(document['embedding'])
    stored_embeddings.append(embedding)


# Compare the new embedding with stored embeddings
for i, stored_embedding in enumerate(stored_embeddings):
    # Reshape embeddings1 to 2D
    embeddings1_2d = embeddings1.reshape(1, -1)
    
    # Reshape stored_embedding to 2D
    stored_embedding_2d = stored_embedding.reshape(1, -1)

    cosine_sim = cosine_similarity(embeddings1_2d, stored_embedding_2d)[0][0]
    print(f"Cosine Similarity with stored embedding {i + 1}: {cosine_sim}")


# Close the connection
client.close()