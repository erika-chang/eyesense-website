import streamlit as st

st.title('Eyesense 👁️')

st.sidebar.success('')
st.text("Eyesense is an AI-based tool to predict eye-related disesases.")
"### How to use Eyesense:"
st.text("Simply upload your image and click 'Predict!' 🧙‍♀️ ")

image_file = st.file_uploader("Upload you image file here:", accept_multiple_files=False, type=['jpeg', 'png', 'jpg'])

if image_file:
    st.image(image_file)

st.button("Predict! 🧙‍♀️")

st.button("Numbers for nerds 🤓")
