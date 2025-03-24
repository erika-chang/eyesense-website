import streamlit as st
import requests

# Configuração da página
st.set_page_config(page_title="Eyesense", page_icon="👁️", layout="wide")

# Criação da barra de navegação como links
st.markdown(
    """
    <style>
        .nav-links {
            display: flex;
            justify-content: center;
            gap: 30px;
            padding: 10px 0;
            background-color: #f8f9fa;
            border-bottom: 2px solid #ddd;
        }
        .nav-links a {
            text-decoration: none;
            color: black;
            font-weight: bold;
            font-size: 18px;
            padding: 10px 15px;
        }
        .nav-links a:hover {
            color: #007bff;
        }
    </style>
    <div class="nav-links">
        <a href="?page=home">Home</a>
        <a href="?page=about-us">About Us</a>
        <a href="?page=about-project">About the Project</a>
        <a href="?page=about-model">About the Model</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Lógica para trocar entre páginas com base no parâmetro da URL
query_params = st.query_params
page = query_params.get("page", "home")

if page == "home":
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

elif page == "about-us":
    st.title("About Us 🧑‍💻")
    st.write("""This project was idealized as part of the Data Science & AI Bootcamp from LeWagon.

        We are four data science enthusiasts with different backgrounds that united for a single purpose: make eye disease diagnosis simpler.

        Thank you for using our tool!

        Claudio, Erika, George and João.""")
    st.image('https://erika-chang.github.io/eyesense_team.png')

elif page == "about-project":
    st.title("About the Project 🚀")
    st.write("""Worldwide, about 2.2 million people have vision impairment. An early and efficient diagnosis tool could prevent about half of those cases.

        That's why we at Eyesense developed an AI-powered tool to help doctors diagnose the most common eye diseases using only one eye fundus image.

        Our models were trained using an ophthalmic database of 5,000 patients (right and left eye) and doctors' diagnostic keywords. This dataset represents patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China.

        The model can classify the image into eight labels:
        -Normal (N)
        -Diabetes (D)
        -Glaucoma (G)
        -Cataract (C)
        -Age-related Macular Degeneration (A)
        -Hypertension (H)
        -Pathological Myopia (M)
        -Other diseases/abnormalities (O)""")

elif page == "about-model":
    st.title("About the Model 🤖")
    st.write("The Eyesense AI model is based on deep learning techniques, trained with thousands of medical images to accurately predict eye conditions.")
