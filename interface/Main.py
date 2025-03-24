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
                        url = 'https://api2-954721262593.europe-west1.run.app/predict'
                        response = requests.post(url, files={'img': img_bytes})


                        if response.status_code == 200:
                            prediction = response.json().get("result", "No result returned")
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
        We are a team of **data enthusiasts** committed to **improving technology**
        for early detection of **eye diseases** and **healthcare accessibility** worldwide.

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
        - **Training Data:** 6392 images (70% training, 17.73% validation, 7.06% test)
        - **Accuracy:** ~92% on test datasets

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
          'Age-related Degeneration (A)', 'Myopia (M)', 'Hypertension (H)']
        sizes = [2873, 1608, 708, 293, 284, 266, 232, 128]
        colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c4e17f']
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0)  # Destacar a maior classe

        # Criando o gráfico de pizza

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%.0f%%', colors=colors, startangle=140, explode=explode)
        ax.axis('equal')  # Garantir que o gráfico seja um círculo

        # Exibir no Streamlit
        st.pyplot(fig)

    st.markdown("""
                The original dataset was unbalanced, so we applied data augmentation techniques to achieve a more even distribution.
                This process involved slight rotations, contrast adjustments, and resizing to enhance variability.

                Additionally, we removed the "Others" category, as it encompassed various eye-related diseases with potentially distinct image patterns.

                After these adjustments, the data distribution was as follows:
                """)

    col1, col2, col3 = st.columns(3)

    with col2:
        # Definição das classes e contagens
        classes = ['Cataract (C)', 'Age-related Degeneration (A)', 'Diabetes (D)', 'Glaucoma (G)', 'Hypertension (H)', 'Myopia (M)', 'Normal (N)']
        counts = [1696, 1536, 1159, 744, 1648, 1344, 2070]  # Contagens correspondentes às categorias

        colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']
        explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1)  # Destaque para cada fatia

        # Criando o gráfico de pizza
        fig, ax = plt.subplots(figsize=(6, 6))  # Define um tamanho adequado
        ax.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=140, explode=explode,
       textprops={'fontsize': 10})  # Define o tamanho das legendas

        ax.axis('equal')  # Mantém o formato circular

        # Exibir no Streamlit
        st.pyplot(fig)

    st.markdown("### 📊 Model's metrics")
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

    st.markdown("""
                The **accuracy** is the portion of correct predictions made from the model in compairson to the overal number of predictions.
                The accuracy is high for diseases as Degeneration (0.92), Hypertension (0.96) e Myopia (0.92).
                For the classification as Normal (0.52) teh accuracy is low, suggesting that the model may have dificulties classifying correctly healthy eyes compared with the diseases.

                The **recall** measures the model's capacity identifying all the true positive instances, i.e., it shows the porcentage of true cases correctly classified as true. A high recall value usually indicates that the model is indicating correctly the presence of a disease.
                Our model has a relatively low recall for diseases such as Cataract (0.47) and Hypertension (0.49). All reacll numbers vary around 0.5, which indicates that approximately 50% of cases are true and classified as true.

                The **F1 score** is the harmonic average between the precision and the recall. It is useful to evaluate the model's performance in terms of balance between the two metrics. The F1 score is specially useful when there is an imbalance between classes.

                Again, the F1 score in all cases is arround 0.5. Even tough we tried balancing the classes with data augmentation, our model did not present a significative improvement in metrics, because the operations made on the images did not improve significative features for the data classification.

                The **precision** is the proportion of positive results predicted correctly as a result of all the instances predicted as positive, i.e. of all the predictions made by the model, how many where correct.
                High precision numbers indicate a high probability that the classification is correct.

                The precision arround 0.5 indicates that approximately 50% of cases are false positive for diseases.

                The **ROC AUC** (ROC area under curve) is a metric that measures the capacity of the model to distinguish between classes. The closer it is to 1, better is the model when separating the classes.
                The area under the curve is the rate of false positives versus true positives.

                Again, the ROC AUC has values around 0.5 indicating that the model has some dificulty in distinguishing between the diseases.

                Altough the model has high accuracy in some eye-related diseases, it still has room for improvement, specially when we take the other metrics into account. The imbalance between classes may play an important role in the model's overall performance, specially in underrepresented classes.
                """)
