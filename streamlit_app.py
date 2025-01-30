# importing required libraries
import streamlit as st
import os
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile
from pymongo import MongoClient
from bson.objectid import ObjectId
from keras.models import load_model
import boto3
import io

# Database setup
client = MongoClient ("mongodb+srv://isiborosasere8:martin2004@cluster2.fjx1hoi.mongodb.net/")
db = client["nigeria_foods_model"]
# accessing the collections of ingredients
collection = db["ingredients"]
# The document's _id
document_id = ObjectId("667ca8a8fccb94110da6b906")

# loading in our model and class names
class_names = ['Abacha (African Salad)',
               'Akara and Eko',
               'Amala and Gbegiri (Ewedu)',
               'Asaro (yam porridge)',
               'Roasted Plantain (bole)',
               'Chin Chin',
               'Egusi Soup',
               'Ewa Agoyin',
               'Fried Plantains (dodo)',
               'Jollof Rice',
               'Meat Pie',
               'moi moi',
               'Nkwobi',
               'Okro Soup',
               'Pepper Soup',
               'Puff Puff',
               'Suya',
               'Vegetable Soup']



# setting the aws parameters
bucket_name = "osas"
local_path = 'nigeria_food_model'
s3_prefix = 'ml-models/'

# code to download our model folder thats been deployed on s3 bucket
s3 = boto3.client('s3')
def download_dir(local_path, s3_prefix):
    s3 = boto3.client('s3')
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                local_file = os.path.join(
                    local_path, os.path.relpath(s3_key, s3_prefix))
                # os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)


st.title("Afrcan Dish classifer Deployment at the Server!!!")

button = st.button("Download Model")
if button:
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)


# File uploader
uploaded_file = st.file_uploader("Choose an African Dish", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        # Read the image
        image = Image.open(uploaded_file)

        # Convert image to uncompressed format (PNG) for better quality
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Display high-quality image
        st.image(img_buffer, width = 400, caption="Uploaded Image")

        # Save to a temporary file and get file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filepath = temp_file.name  # Get the file path

#predict button
predict = st.button("Predict")

# a function that makes prediction
def predict_label (img_path):
    """
    A function that read, processes and make predictions on our custom images
    """
    
    MODEL_PATH = "nigeria_food_model/nigeria_food_model_efficientNetB3.h5"
    model = load_model(MODEL_PATH)
    
    # read in the image file and preprocess it
    img = tf.io.read_file (img_path)
    img = tf.image.decode_image (img)
    img = tf.image.resize (img, (224, 224))
    
    
    # making predictions
    prediction = model.predict (tf.expand_dims (img, axis = 0))
    pred_class = class_names[prediction.argmax ()]
    
    return pred_class

# a function that gets the ingredients
# Function to get ingredients from MongoDB
def get_ingredients(prediction):
    # Ensure document_id is correctly retrieved (Modify this as per your DB structure)
    result = collection.find_one({"_id": document_id})  
    
    if not result:
        st.error("Error: No matching document found in the database.")
        return

    if prediction == "Akara and Eko":
        akara_and_eko = result["Akara and Eko"]
        st.subheader("AKARA")
        for ingredient in akara_and_eko["Akara"]:
            st.write(ingredient)
        st.subheader("EKO")
        for ingredient in akara_and_eko["Eko"]:
            st.write(ingredient)

    elif prediction == "Amala and Gbegiri (Ewedu)":
        amala_and_gbegiri = result["Amala and Gbegiri (Ewedu)"]
        st.subheader("AMALA")
        for ingredient in amala_and_gbegiri["Amala"]:
            st.write(ingredient)  
        st.subheader("GBEGIRI")
        for ingredient in amala_and_gbegiri["Gbegiri"]:
            st.write(ingredient)

    elif prediction == "Meat Pie":
        meat_pie = result["Meat Pie"]
        st.subheader("DOUGH")
        for ingredient in meat_pie["dough"]:
            st.write(ingredient) 
        st.subheader("MEAT PIE FILLINGS")
        for ingredient in meat_pie["Meat Pie filling"]:
            st.write(ingredient)

    else:
        ingredients = result.get(prediction, [])
        if not ingredients:
            st.warning("No ingredients found for this dish.")
        else:
            st.subheader(f"Ingredients for {prediction}")
            for ingredient in ingredients:
                st.write(ingredient)


# Ensure prediction happens only when button is clicked
if predict:
    with st.spinner("Predicting..."):
        output = predict_label(temp_filepath)
        st.write(f"**Prediction:** {output}")

        # ✅ Call function directly instead of returning None
        get_ingredients(output)
