# EyeSense: Eye Disease Prediction from Fundus Images

**EyeSense** is an interactive web application that predicts potential eye diseases based on eye-fundus exam images. Built with [Streamlit](https://streamlit.io/), it offers researchers, clinicians, and developers an intuitive platform to assess the probability of various eye impairments using deep learning models.

---

## 🌐 Live Demo

Try the application here:  
🔗 [https://eyesense.streamlit.app/](https://eyesense.streamlit.app/)

---

## 📁 Repository Overview

This repository contains the source code for the EyeSense web app. The project structure is as follows:

- **`interface/`**: Main application scripts, including the Streamlit front end and model integration.
- **`.devcontainer/`**: Development environment configurations for VS Code remote containers.
- **`.python-version`**: Defines the Python version used in the project.

---

## 🚀 Features

- **Simple Image Input**: Upload eye-fundus images with ease to get instant results.
- **Probability-Based Predictions**: The app not only identifies the most likely eye disease but also displays the probability scores for each potential condition.
- **User-Friendly Interface**: Clean and intuitive design, accessible even to non-technical users.
- **AI-Powered Analysis**: Leverages a trained machine learning model to interpret input images.

---

## 🛠️ Installation & Setup

To run the app locally:

### 1. Clone the Repository

```bash
git clone https://github.com/erika-chang/eyesense-website.git
cd eyesense-website
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run interface/app.py
```

Open your browser and navigate to: http://localhost:8501

---

## 📊 Sample Data

Explore the app’s capabilities using sample images:

Navigate to the sample_data/ directory.

Upload the example images in the app to view predictions and model outputs.

---

## 📬 Contact

For questions, feedback, or collaboration inquiries:

Author: Erika Chang de Azevedo
Email: erikaa.chang@gmail.com
LinkedIn: [Erika Chang de Azevedo] (www.linkedin.com/in/ecdazevedo)

