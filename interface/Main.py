import streamlit as st
import requests

# Configuração da página
st.set_page_config(page_title="Eyesense", page_icon="👁️", layout="wide")

# Barra de navegação superior
menu = st.radio("Navigation", ["Home", "About us", "About the project", "About the model"], horizontal=True)

if menu == "Home":
    st.title("Eyesense 👁️")

    # Descrição
    st.markdown("### AI-powered Eye Disease Prediction")
    st.write("Upload an image of an eye, and our AI model will predict potential eye-related diseases.")

    # Upload de imagem
    image_file = st.file_uploader("Upload your image file here:", type=["jpeg", "png", "jpg"])

    # Verifica se o usuário fez upload de uma imagem
    if image_file:
        col1, col2 = st.columns(2)

        with col1:
            # Exibe a imagem carregada
            st.image(image_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.info("Click 'Predict!' to analyze the image.")

        # Botão de previsão
        if st.button("Predict! 🧙‍♀️"):
            with st.spinner("Analyzing... 🔍"):
                try:
                    img_bytes = image_file.getvalue()
                    url = "https://your-api-endpoint.com/predict"  # Substitua pela URL correta
                    response = requests.post(url, files={"file": img_bytes})

                    if response.status_code == 200:
                        prediction = response.json().get("prediction", "No result returned")
                        st.success(f"Prediction: {prediction}")
                    else:
                        st.error("Error fetching prediction. Please try again.")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

elif menu == "About us":
    st.title("About Us 🧑‍💻")
    st.write("We are a team of AI enthusiasts committed to building innovative solutions for healthcare.")

elif menu == "About the project":
    st.title("About the Project 🚀")
    st.write("Eyesense is an AI-driven platform designed to help detect eye-related diseases from images.")

elif menu == "About the model":
    st.title("About the Model 🤖")
    st.write("The Eyesense AI model is based on deep learning techniques, trained with thousands of medical images to accurately predict eye conditions.")
