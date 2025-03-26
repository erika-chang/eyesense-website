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
    if st.button("📌 About the Project", type="tertiary"):
        change_page("about-project")
    if st.button("🤖 About the Model", type="tertiary"):
        change_page("about-model")
    if st.button("👥 About Us", type="tertiary"):
        change_page("about-us")

# Exibição do conteúdo da página selecionada
if st.session_state.current_page == "home":
    st.title("Eyesense 👁️ – AI for Eye Disease Detection")

    st.write(
        "An **AI-powered tool** designed to assist in the early detection of **eye diseases** "
        "using deep learning models trained on thousands of medical images."
    )

    st.markdown("### 🔬 How does it work?")
    #st.write("**Simply upload a fundus eye image**, and our AI model—powered by state-of-the-art deep learning techniques—will analyze it to identify potential eye diseases.")
    st.write("Simply upload your **eye fundus exam**, and our AI model will analyze it for potential diseases.")

    # Upload de imagem
    image_file = st.file_uploader("Upload your image file here:", type=["jpeg", "png", "jpg"])

    # Verifica se o usuário fez upload de uma imagem
    if image_file:
        col1, col2 = st.columns(2)

        with col2:
            st.info("Click 'Predict!' to analyze the image.")
            # Botão de previsão
            if st.button("Predict! 🤖"):
                with st.spinner("Analyzing... 🔍"):
                    try:
                        img_bytes = image_file.getvalue()
                        url = 'https://apieyesense-954721262593.europe-west1.run.app/predict'
                        response = requests.post(url, files={'img': img_bytes})


                        if response.status_code == 200:
                            prediction = response.json().get("result", "No result returned")
                            #st.success(f"Probabilities per class: {prediction}")
                            predict_df = pd.DataFrame(prediction)
                            predict_df.rename(columns={0: "Class", 1:"Probability"}, inplace=True)
                            predict_df['Probability'] = predict_df['Probability'].apply(lambda x: str(round(x*100,2))+"%")
                            st.success(f"The most probable class is: {prediction[0][0].capitalize()}")
                            st.dataframe(predict_df, hide_index=True)

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

        **Eysense** is an advanced **AI-driven technology** designed for the **analysis of fundus eye exams to detect ocular diseases with high precision**.

       By leveraging deep learning, our system assists healthcare professionals in making faster and more accurate diagnoses, enabling early detection and improved patient outcomes.

       With cutting-edge artificial intelligence, Eyesense enhances diagnostic accuracy, supports clinical decision-making, and contributes to the future of ophthalmology.

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

        Worldwide, about 2.2 billion people have vision impairment ([WHO](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)). An early and efficient diagnosis tool could prevent about half of those cases.


        ### 🎯 Project Goals
        - Develop an **accurate AI model** to predict eye diseases from medical images
        - Provide a **user-friendly interface** for doctors and patients
        - Increase **awareness** about eye health and **prevent blindness**

        That's why we at Eyesense developed an AI-powered tool to help doctors diagnose the most common eye diseases using only one eye fundus image.

        The model can classify the image into seven labels:
        - Normal (N)
        - Diabetes (D)
        - Glaucoma (G)
        - Cataract (C)
        - Age-related Macular Degeneration (A)
        - Hypertension (H)
        - Pathological Myopia (M)""")
    st.image("https://erika-chang.github.io/eye_diseases_effetc.png", caption="How different eye diseases affect the vision", use_container_width=True)
    st.markdown("""
        ### 🔍 How We Built It
        - **Deep Learning Models** trained on an ophthalmic database of approximately **5,000 patients**
        - The images were classified according to doctors' diagnostic keywords (information collected by Shanggong Medical Technology Co., Ltd.)
        - **Secure cloud-based API** for fast and reliable predictions
        - Database source: [kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)
        """)
    st.image("https://erika-chang.github.io/flow_site_new_2.png", caption="Schema of how our tool works and main libraries used", use_container_width=True)

elif st.session_state.current_page == "about-model":
    st.title("🤖 About the Model")

    st.markdown(
        """
        ### 🔥 AI Model Overview
        - **Architecture:** Convolutional Neural Networks (CNN), based on **Xception**
        - **Training Data:** 6392 images
        - **Accuracy:** ~92% on test datasets""")

    st.image("https://erika-chang.github.io/xception_build.png", caption="Example of a Xception architecture", use_container_width=True)

    st.markdown(
        """
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
        - Diabetes
        - Glaucoma
        - Hypertension
        - Pathological Myopia

        ### 📚 Technical Details
        - **Frameworks:** TensorFlow
        - **Cloud Hosting:** Google Cloud
        """)

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
                The **accuracy** represents the proportion of correct predictions made by the model in comparison to the total number of predictions. The accuracy is high for diseases such as Degeneration (0.92), Hypertension (0.96), and Myopia (0.92). However, for the Normal classification (0.52), accuracy is significantly lower, suggesting that the model struggles to correctly classify healthy eyes compared to diseased ones.

                The **recall** measures the model's ability to identify all true positive instances, i.e., it indicates the percentage of actual cases correctly classified as positive. A high recall value usually means the model is effective at detecting the presence of a disease. Our model shows relatively low recall for Cataract (0.47) and Hypertension (0.49). In general, recall values hover around 0.5, meaning the model correctly identifies only about 50% of actual cases.

                The **F1 score** is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. This metric is particularly useful in cases of class imbalance.
                Again, the F1 scores across all classes remain around 0.5. Despite applying data augmentation to balance the dataset, the model did not show significant improvements in these metrics. This suggests that the transformations applied to the images did not enhance distinguishable features for classification.

                The **precision** metric measures how many of the positive predictions made by the model were actually correct. A high precision score indicates that when the model classifies an instance as positive, it is likely to be correct.
                With precision values around 0.5, the model exhibits a high rate of false positives, meaning that approximately 50% of cases predicted as diseased are actually incorrect.

                The **ROC AUC** (Receiver Operating Characteristic - Area Under Curve) evaluates the model's ability to distinguish between different classes. A value closer to 1.0 indicates a strong capability to separate classes, while values around 0.5 suggest poor differentiation. In this case, the ROC AUC values remain close to 0.5, indicating the model struggles to effectively differentiate between diseases.

                Although the model demonstrates high accuracy for some eye-related diseases, there is still significant room for improvement, especially when considering the other evaluation metrics. Class imbalance may be a key factor affecting overall performance, particularly for underrepresented categories. Addressing this imbalance with more robust data augmentation techniques, additional labeled data, or improved feature extraction could help enhance the model’s classification abilities.
                """)

    st.markdown("""
                ### 🚀 Xception Under the Hood
                Xception, developed by Google, stands for **Extreme version of Inception**. It introduces a modified depthwise separable convolution, outperforming Inception-v3 (Google’s 1st Runner-Up in ILSVRC 2015) on both ImageNet ILSVRC and JFT datasets.

                🔎 What’s Covered?
                - Original Depthwise Separable Convolution
                - Modified Depthwise Separable Convolution in Xception
                - Overall Architecture
                - Comparison with State-of-the-Art Results

                1. Original Depthwise Separable Convolution""")

    st.image("https://erika-chang.github.io/Depthwise Convolution.webp", caption="Original Depthwise Separable Convolution architecture", width= 600)

    st.markdown("""
                The original depthwise separable convolution consists of:
                -  Depthwise convolution – A channel-wise n×n spatial convolution. If there are 5 channels, then 5 separate n×n spatial convolutions are performed.
                - Pointwise convolution – A 1×1 convolution that adjusts the dimensionality.

                💡 Why is this better than conventional convolution?

                Unlike traditional convolution, depthwise separable convolution reduces the number of connections, making the model lighter and more efficient.

                2. Modified Depthwise Separable Convolution in Xception""")

    st.image("https://erika-chang.github.io/Pointwise Convolution.webp", caption=" Modified Depthwise Separable Convolution architecture in Xception", width= 600)

    st.markdown("""
                In Xception, the depthwise separable convolution is rearranged:
                - Pointwise convolution (1×1) is applied first
                - Depthwise convolution (n×n) follows after

                💡 Why this modification?
                This design is inspired by the Inception module in Inception-v3, where 1×1 convolutions are applied before any n×n spatial convolutions (typically 3×3).

                🔬 Key Differences from the Original Version
                - Order of Operations
                Original: Depthwise convolution → 1×1 convolution
                Xception: 1×1 convolution → Depthwise convolution
                However, this change is considered minor when using multiple stacked layers.


                - Non-Linearity
                In Inception, a ReLU activation is applied after the first operation.
                In Xception, there is NO intermediate activation function (ReLU/ELU), leading to higher accuracy.

                - Performance Comparison
                Xception without intermediate activation functions achieves higher accuracy than versions using ELU or ReLU.""")

    st.image("https://erika-chang.github.io/No intermediate activation.webp", caption="Performance Comparison: no  intermediate activation x ELU x ReLU", width= 600)

    st.markdown("""
                3. Overall Architecture
                Xception treats SeparableConv (modified depthwise separable convolution) as Inception Modules, integrating them throughout the deep learning architecture.
                - Residual (shortcut/skip) connections (inspired by ResNet) are incorporated in all flows.

                  Residual Connections Matter!

                  ✅ With residual connections → Higher accuracy

                  ❌ Without residual connections → Lower accuracy

                  📈 Xception performs significantly better with residual connections, proving their importance in deep learning architectures!

                Xception's optimized depthwise separable convolution and residual connections make it one of the most powerful architectures for image classification!

                Reference: [Chollet, F., Xception: Deep Learning with Depthwise Separable Convolutions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
                """)
