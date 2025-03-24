import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

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
        - **Architecture:** Convolutional Neural Networks (CNN), based on **Xception**
        - **Training Data:** 6392 images
        - **Accuracy:** ~92% on validation datasets

        ### 📚 Model Description
        The **Eyesense model** uses the powerful **Xception** architecture, a deep learning model known for its **efficiency** and **high performance** in image classification tasks.

        - **Base Model:** The model uses the pre-trained **Xception network** with weights from ImageNet, a large dataset of labeled images. This helps the model generalize better to unseen images.

        - **Freezing Layers:** The base model's layers are **frozen** to prevent them from being updated during training. This allows the model to retain the features learned from ImageNet.

        - **Global Average Pooling:** Instead of using fully connected layers, **Global Average Pooling** is applied to reduce the spatial dimensions of the output, which improves efficiency and reduces the risk of overfitting.

        - **Dropout Regularization:** Dropout is applied to certain layers to regularize the model, preventing it from overfitting on the training data.

        - **Fully Connected Layer:** After pooling, a dense layer with **1024 units** and **ReLU activation** is added to help the model learn complex representations from the data.

        - **Output Layer:** The model ends with a softmax output layer with **7 units**, one for each disease class (cataract, degeneration, diabetes, glaucoma, hypertension, myopia, and normal).

        ### 🏥 Diseases Detected
        - Cataract
        - Age-related Macular Degeneration
        - Diabets
        - Glaucoma
        - Hypertension
        - Pathological Myopia

        ### 📚 Technical Details
        - **Frameworks:** TensorFlow
        - **Cloud Hosting:** Google Cloud
        """
    )
    st.markdown("### 📊 Data distribution")
    # Dados das classes
    col1, col2, col3 = st.columns(3)

    with col2:
        labels = ['Normal (N)', 'Diabetes (D)', 'Other (O)', 'Cataract (C)', 'Glaucoma (G)',
          'Age-related (A)', 'Myopia (M)', 'Hypertension (H)']
        sizes = [2873, 1608, 708, 293, 284, 266, 232, 128]
        colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c4e17f']
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0)  # Destacar a maior classe

        # Criando o gráfico de pizza

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%.0f%%', colors=colors, startangle=140, explode=explode)
        ax.axis('equal')  # Garantir que o gráfico seja um círculo

        # Exibir no Streamlit
        st.pyplot(fig)

    st.markdown(""""
                The original dataset was unbalanced, so we applied data augmentation techniques to improve its distribution.
                The augmentation process included slight rotations, adjustments in image contrast, and resizing.
                After applying these enhancements, the data distribution was as follows:
                """)

    col1, col2, col3 = st.columns(3)

    with col2:
        # Definição das classes e contagens
        classes = ['Cataract', 'Degeneration', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal']
        counts = [1696, 1536, 1159, 744, 1648, 1344, 2070]  # Contagens correspondentes às categorias

        colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']
        explode = (0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05)  # Destaque para cada fatia

        # Criando o gráfico de pizza
        fig, ax = plt.subplots(figsize=(6, 6))  # Define um tamanho adequado
        ax.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=140, explode=explode,
       textprops={'fontsize': 10})  # Define o tamanho das legendas

        ax.axis('equal')  # Mantém o formato circular

        # Exibir no Streamlit
        st.pyplot(fig)

    # Dados da tabela
    data = {
        "Class": ["Cataract", "Degeneration", "Diabetes", "Glaucoma", "Hypertension", "Myopia", "Normal"],
        "Accuracy": [0.89, 0.92, 0.69, 0.87, 0.96, 0.92, 0.52],
        "Recall": [0.47, 0.53, 0.53, 0.52, 0.49, 0.51, 0.52],
        "F1 Score": [0.47, 0.53, 0.51, 0.51, 0.49, 0.51, 0.51],
        "Precision": [0.47, 0.53, 0.56, 0.51, 0.49, 0.51, 0.52],
        "ROC AUC": [0.47, 0.53, 0.53, 0.52, 0.49, 0.51, 0.52],
    }
    # Criando um DataFrame
    df = pd.DataFrame(data)
    df.set_index("Class", inplace=True)


    # Exibindo a tabela
    st.dataframe(df)

    st.markdown("""" """)
