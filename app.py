import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("room_model.keras")

# Labels (adjust based on your training folder order)
labels = ["No Maintenance Needed", "Maintenance Needed"]

st.title("Room Maintenance Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))  # same as training size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # ✅ normalize exactly like training
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Case 1: binary (1 output neuron with sigmoid)
    if prediction.shape[1] == 1:
        pred_value = (prediction[0][0] > 0.5).astype(int)  
        result = labels[pred_value]

    # Case 2: categorical (2 output neurons with softmax)
    else:
        pred_class = np.argmax(prediction, axis=1)[0]
        result = labels[pred_class]

    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.success(f"Prediction: {result}")
    st.write("Raw prediction:", prediction)
