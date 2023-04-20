import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define function to scale down the image
def scale_image(image):
    # Scale the image to 224x224 pixels
    scaled_image = image.resize((512, 512))
    # Convert the image to a NumPy array
    scaled_image_array = np.array(scaled_image)
    # Reshape the array to have a batch size of 1
    scaled_image_array = np.expand_dims(scaled_image_array, axis=0)
    # Normalize the pixel values to be between 0 and 1
    scaled_image_array = scaled_image_array / 255.0
    return scaled_image_array
 
# Define function to predict the cyclone intensity
def predict_cyclone_intensity(image):
    # Load the pre-trained AI model
    model = tf.keras.models.load_model('Model.h5')
    # Scale down the image
    scaled_image = scale_image(image)
    # Make the prediction
    prediction = model.predict(scaled_image)[0][0]
    # Return the prediction
    return prediction

# Set up the Streamlit page
st.set_page_config(page_title='Cyclone Intensity Prediction', page_icon=':cyclone:')
st.title('Cyclone Intensity Prediction')
st.markdown('Upload an image of a cyclone to predict its intensity.')
uploaded_image = st.file_uploader('', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Scale down the image and make the prediction
    scaled_image = scale_image(image)
    prediction = predict_cyclone_intensity(image)
    # Display the prediction
    st.markdown('## Cyclone Intensity Prediction')
    st.write(f'The predicted intensity of the cyclone is {prediction:.2f}.')
    # Add a graph showing the predicted intensity over time
    st.markdown('## Cyclone Intensity Over Time')
    intensity_data = np.random.rand(10) * 5 # Placeholder data
    st.line_chart(intensity_data)
    # Add a feature to compare multiple cyclones
    st.markdown('## Compare Cyclones')
    uploaded_images = st.file_uploader('Upload images of cyclones to compare', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if uploaded_images is not None:
        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image)
            st.image(image, caption=uploaded_image.name, use_column_width=True)
            prediction = predict_cyclone_intensity(image)
            st.write(f'The predicted intensity of {uploaded_image.name} is {prediction:.2f}.')
    # Add social media sharing buttons
    st.markdown('## Share Prediction')
    share_url = 'https://my-cyclone-app.com/prediction' # Placeholder URL
    st.write(f'Share your prediction: {share_url}')
   
