import streamlit as st
import requests

st.title('Eyesense 👁️')

st.sidebar.success('')
st.text("Eyesense is an AI-based tool to predict eye-related disesases.")
"### How to use Eyesense:"
st.text("Simply upload your image and click 'Predict!' 🧙‍♀️ ")

image_file = st.file_uploader("Upload you image file here:", accept_multiple_files=False, type=['jpeg', 'png', 'jpg'])

if image_file:
    col1, col2 = st.columns(2)

    with col1:
    ### Display the image user uploaded
        st.image(image_file, caption="Here's the image you uploaded ☝️")

    with col2:
        with st.spinner("Wait for it..."):
        ### Get bytes from the file buffer
            img_bytes = image_file.getvalue()


if st.button("Predict! 🧙‍♀️"):
    url = 'https://taxifare.lewagon.ai/predict'
    response = requests.post(url + "/predict", files={'img': img_bytes})
    if response.status_code == 200:
        prediction = response.json().get("result", "Error: No fare returned")
        st.success(f"Estimated Fare: ${prediction:.2f}")
    else:
        st.error("Error fetching prediction. Please try again.")


st.button("Numbers for nerds 🤓")
