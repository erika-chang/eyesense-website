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
    if st.button("🏠 Home", type="tertiary"):
        change_page("home")
    if st.button("👥 About Us", type="tertiary"):
        change_page("about-us")
    if st.button("📌 About the Project", type="tertiary"):
        change_page("about-project")
    if st.button("🤖 About the Model", type="tertiary"):
        change_page("about-model")

# Exibição do conteúdo da página selecionada
if st.session_state.current_page == "home":
    st.title("Eyesense 👁️ – AI for Eye Disease Detection")
    st.write(
        "Eyesense is an **AI-powered tool** designed to assist in the early detection of **eye diseases** "
        "using deep learning models trained on thousands of medical images."
    )

    st.markdown("### 🔬 How does it work?")
    st.write("Simply **upload an image** of an eye, and our AI model will analyze it for potential diseases.")

    # Upload de imagem
    image_file = st.file_uploader("Upload your image file here:", type=["jpeg", "png", "jpg"])

    # Verifica se o usuário fez upload de uma imagem
    if image_file:
        col1, col2 = st.columns(2)

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

        with col1:
            # Exibe a imagem carregada
            st.image(image_file, caption="Uploaded Image", use_container_width=True)

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
        This project was idealized as part of the Data Science & AI Bootcamp from LeWagon.

        Worldwide, about 2.2 million people have vision impairment. An early and efficient diagnosis tool could prevent about half of those cases.


        ### 🎯 Project Goals
        - Develop an **accurate AI model** to predict eye diseases from medical images
        - Provide a **user-friendly interface** for doctors and patients
        - Increase **awareness** about eye health and **prevent blindness**
        That's why we at Eyesense developed an AI-powered tool to help doctors diagnose the most common eye diseases using only one eye fundus image.

        The model can classify the image into eight labels:
        - Normal (N)
        - Diabetes (D)
        - Glaucoma (G)
        - Cataract (C)
        - Age-related Macular Degeneration (A)
        - Hypertension (H)
        - Pathological Myopia (M)
        - Other diseases/abnormalities (O)

        ### 🔍 How We Built It
        - **Deep Learning Models** trained on an ophthalmic database of **5,000 patients**
        - The images were classified according to doctors' diagnostic keywords (information collected by Shanggong Medical Technology Co., Ltd.)
        - **Secure cloud-based API** for fast and reliable predictions
        - Database source: [kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)
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

        ### 🏥 Diseases detected
        - Diabetes
        - Glaucoma
        - Cataract
        - Age-related Macular Degeneration
        - Hypertension
        - Pathological Myopia
        - Other diseases/abnormalities

        ### 📚 Technical Details
        - **Frameworks:** TensorFlow, PyTorch
        - **Cloud Hosting:** AWS & Google Cloud
        - **Model Version:** v1.2 (updated regularly)
        """
    )
