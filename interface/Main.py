import streamlit as st
import requests

# Configuração da página
st.set_page_config(page_title="Eyesense", page_icon="👁️", layout="wide")

# Inicializa session_state para armazenar a página ativa
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Função para mudar de página sem recarregar
def change_page(page):
    st.session_state.current_page = page

# Barra lateral para navegação
with st.sidebar:
    st.title("🔍 Eyesense Navigation")
    if st.button("🏠 Home"):
        change_page("home")
    if st.button("👥 About Us"):
        change_page("about-us")
    if st.button("📌 About the Project"):
        change_page("about-project")
    if st.button("🤖 About the Model"):
        change_page("about-model")

# Exibição do conteúdo da página selecionada
if st.session_state.current_page == "home":
    st.title("Eyesense 👁️")

    # Descrição
    st.markdown("### AI-powered Eye Disease Prediction")
    st.write("Upload an image of an eye, and our AI model will predict potential eye-related diseases.")

    # Upload de imagem
    image_file = st.file_uploader("Upload your image file here:", type=["jpeg", "png", "jpg"])

    # Verifica se o usuário fez upload de uma imagem
    if image_file:

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

        # Exibe a imagem carregada
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

elif st.session_state.current_page == "about-us":
    st.title("👥 About Us")
    st.markdown(
        """
        ### 🌎 Our Mission
        We are a team of **data enthusiasts** committed to **leveraging technology**
        for early detection of **eye diseases** and **improving healthcare accessibility** worldwide.

        ### 🏆 Why Eyesense?
        - 🚀 Cutting-edge AI technology
        - 🏥 Support for medical professionals
        - 🌍 Global health impact

        ### 📱 Get in Touch with Us:
        - [LinkedIn: Claudio](https://www.linkedin.com/in/caazzi)
        - [LinkedIn: Erika](https://www.linkedin.com/in/ecdazevedo)
        - [LinkedIn: George](https://www.linkedin.com/in/george-silva-448a7321/)
        - [GitHub: João](https://github.com/masalesvic)
        """
    )

elif st.session_state.current_page == "about-project":
    st.title("📌 About the Project")
    st.markdown(
        """
        ### 🎯 Project Goals
        - Develop an **accurate AI model** to predict eye diseases from medical images
        - Provide a **user-friendly interface** for doctors and patients
        - Increase **awareness** about eye health and **prevent blindness**

        ### 🔍 How We Built It
        - **Deep Learning Models** trained on **thousands** of labeled images
        - **Collaboration with ophthalmologists** to refine accuracy
        - **Secure cloud-based API** for fast and reliable predictions
        """
    )

elif st.session_state.current_page == "about-model":
    st.title("🤖 About the Model")
    st.markdown(
        """
        ### 🔥 AI Model Overview
        - **Architecture:** Convolutional Neural Networks (CNN)
        - **Training Data:** Thousands of **real medical images**
        - **Accuracy:** ~92% on validation datasets

        ### 🏥 Diseases Detected
        - Glaucoma
        - Diabetic Retinopathy
        - Cataracts
        - Other common eye conditions

        ### 📚 Technical Details
        - **Frameworks:** TensorFlow, PyTorch
        - **Cloud Hosting:** AWS & Google Cloud
        - **Model Version:** v1.2 (updated regularly)
        """
    )
