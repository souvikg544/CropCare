import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


st.set_page_config(
   page_title="CropCare",
 page_icon="crops.jpg",
 initial_sidebar_state="expanded")

#@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def load(s):
    if(s=="potato"):
        with st.spinner("Potato Model is loading"):
            model = tf.keras.models.load_model("models/model111")
        return model
    elif(s=="tomato"):
        with st.spinner("Tomato Model is loading"):
            model = tf.keras.models.load_model("models/tomatomodel1.h5")
        return model
    else:
        return 0

def potato():
    st.title("Potato Disease Prediction")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpeg', 'jpg','jfif'],key="2")
    cn=["Early Blight/अगेंती अंगमारी","Late Blight/उत्तरभावी अंगमारी","Healthy/स्वस्थ"]
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image',width=400)
        array = tf.keras.preprocessing.image.img_to_array(image)

        if(st.button("See prediction")):
            model1 = load("potato")
            with st.spinner("Prediction may take some time"):
                prediction = model1.predict(np.expand_dims(array, 0))
            col1,col2 = st.columns(2)
            predicted_class = cn[np.argmax(prediction[0])]
            confidence = round(100 * (np.max(prediction[0])), 2)
            if(prediction[0][2]>0.20):
                predicted_class = "Healthy/स्वस्थ"
                confidence = round(100 * prediction[0][2], 2)

            with col1:
                st.subheader("Disease")
                st.success(predicted_class)
            with col2:
                st.subheader("Prediction Confidence")
                st.info(confidence)
                #st.write(prediction[0])

def tomato():
    st.title("Tomato Disease Prediction")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpeg', 'jpg','jfif'],key="1")
    cn=['Bacterial spot','Early blight','Late blight', 'Leaf Mold','Septoria leaf spot',
          'Spider mites Two-spotted spider mite','Target Spot', 'Yellow Leaf Curl Virus',
             'mosaic virus','healthy']
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image',width=400)
        array = tf.keras.preprocessing.image.img_to_array(image)

        if(st.button("See prediction")):
            model1 = load("tomato")
            with st.spinner("Prediction may take some time"):
                prediction = model1.predict(np.expand_dims(array, 0))
            col1,col2 = st.columns(2)
            predicted_class = cn[np.argmax(prediction[0])]
            confidence = round(100 * (np.max(prediction[0])), 2)

            with col1:
                st.subheader("Disease")
                st.success(predicted_class)
            with col2:
                st.subheader("Prediction Confidence")
                st.info(confidence)
                #st.write(prediction[0])
def pepper():
    st.title("Pepper Disease Prediction")
    st.info("Model is still under progress")

st.sidebar.title("Plant Diseases")
pages=st.sidebar.radio(label="",options=["Potato","Tomato","Pepper"])
if(pages=="Potato"):
    potato()
elif(pages=="Tomato"):
    tomato()
else:
    pepper()
st.sidebar.markdown("----------------------")
st.sidebar.subheader("Developed By Souvik")
st.sidebar.markdown("----------------------")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/souvik-ghosh-3b8b411b2/)")
st.sidebar.write("@souvikg544@gmail.com")


