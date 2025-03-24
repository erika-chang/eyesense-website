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

# Estilo para melhorar a sidebar
st.markdown(
    """
    <style>
        .sidebar-text {
            font-size: 18px;
        }
        .sidebar-link {
            text-decoration: none;
            font-weight: bold;
            color: #007bff;
            font-size: 20px;
        }
        .sidebar-link:hover {
            color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Barra lateral para navegação com links estilizados
with st.sidebar:
    st.title("🔍 Eyesense Navigation")
    st.markdown('<p class="sidebar-text">Navigate through the sections:</p>', unsafe_allow_html=True)
    st.markdown('<a href="#" class="sidebar-link" onclick="window.location.hash=\'home\'">🏠 Home</a>', unsafe_allow_html=True)
    st.markdown('<a href="#" class="sidebar-link" onclick="window.location.hash=\'about-us\'">👥 About Us</a>', unsafe_allow_html=True)
    st.markdown('<a href="#" class="sidebar-link" onclick="window.location.hash=\'about-project\'">📌 About the Project</a>', unsafe_allow_html=True)
    st.markdown('<a href="#" class="sidebar-link" onclick="window.location.hash=\'about-model\'">🤖 About the Model</a>', unsafe_allow_html=True)

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
    image_file = st.file_uploader("📤 Upload your image file here:", type=["jpeg", "png", "jpg"])

    # Verifica se o usuário fez upload de uma imagem
    if image_file:
        col1, col2 = st.columns(2)

        with col1:
            # Exibe a imagem carregada
            st.image(image_file, caption="👁️ Uploaded Image", use_column_width=True)

        with col2:
            st.info("Click 'Predict!' to analyze the image.")

        # Botão de previsão
        if st.button("🔍 Predict!"):
            with st.spinner("Analyzing... 🕵️‍♂️"):
                try:
                    img_bytes = image_file.getvalue()
                    url = "https://your-api-endpoint.com/predict"  # Substitua pela URL correta
                    response = requests.post(url, files={"file": img_bytes})

                    if response.status_code == 200:
                        prediction = response.json().get("prediction", "No result returned")
                        st.success(f"🩺 Prediction: **{prediction}**")
                    else:
                        st.error("⚠️ Error fetching prediction. Please try again.")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

elif st.session_state.current_page == "about-us":
    st.title("👥 About Us")
    st.markdown(
        """
        ### 🌎 Our Mission
        We are a team of **AI researchers, medical professionals, and developers** committed to **leveraging technology**
        for early detection of **eye diseases** and **improving healthcare accessibility** worldwide.

        ### 🏆 Why Eyesense?
        - 🚀 Cutting-edge AI technology
        - 🏥 Support for medical professionals
        - 🌍 Global health impact

        For more information, contact us at **[contact@eyesense.com](mailto:contact@eyesense.com)**
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
