import streamlit as st
import requests

# Configuração da página
st.set_page_config(page_title="Eyesense", page_icon="👁️", layout="centered")

# Título
st.title("Eyesense 👁️")

# Barra lateral
st.sidebar.success("Upload an image and let AI do the magic!")

# Descrição
st.markdown("### Eyesense: AI-powered Eye Disease Prediction")
st.write("Upload an image of an eye, and our AI model will predict potential eye-related diseases.")

# Upload de imagem
image_file = st.file_uploader("Upload your image file here:", type=["jpeg", "png", "jpg"])

# Verifica se o usuário fez upload de uma imagem
if image_file:
    col1, col2 = st.columns(2)

    with col1:
        # Exibe a imagem carregada
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

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

# Botão extra (exemplo)
if st.button("Numbers for nerds 🤓"):
    st.write("Coming soon: Detailed analysis and statistics!")
