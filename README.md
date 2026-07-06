# 💳 Credit Card Fraud Detection using Hybrid Machine Learning

A **Streamlit-based Machine Learning application** that predicts whether a credit card transaction is likely to be **fraudulent or legitimate** using a **hybrid AI pipeline**.

Instead of relying on a single machine learning model, this project combines **Deep Learning** and **Traditional Machine Learning** to improve feature representation and classification performance.

The workflow consists of:

- **Autoencoder (TensorFlow/Keras)** for learning compressed latent representations of transaction data.
- **Random Forest Classifier (Scikit-learn)** that utilizes both the encoded features and contextual transaction features for final fraud prediction.

The application provides an easy-to-use web interface built with **Streamlit**, making it suitable for demonstrations, portfolio projects, and learning purposes.

---

# 🚀 Features

- Interactive Streamlit web application
- Hybrid Deep Learning + Machine Learning architecture
- Autoencoder-based feature extraction
- Random Forest fraud classification
- Clean and responsive UI
- Fast prediction with pre-trained models
- Modular project structure
- Separate training and inference pipelines
- Model persistence using Joblib and TensorFlow
- Easily extensible for real-world fraud detection systems

---

# 🏗️ Project Architecture

```
Raw Transaction Features
            │
            ▼
     Data Preprocessing
            │
            ▼
     Autoencoder Encoder
            │
     Encoded Features
            │
            ├───────────────┐
            │               │
            ▼               ▼
 Contextual Features   Encoded Features
            │               │
            └───────┬───────┘
                    ▼
         Feature Concatenation
                    ▼
        Random Forest Classifier
                    ▼
      Fraud / Legitimate Prediction
```

---

# 📂 Project Structure

```
.
├── app.py                         # Streamlit web application
├── utils.py                       # Model loading and prediction utilities
├── train_and_save.py              # Training pipeline
│
├── models/
│   ├── encoder_model.h5
│   ├── hybrid_scaler.joblib
│   ├── rf_hybrid_model.joblib
│   └── feature_means.csv
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis
│
├── reports/
│   └── training_report.txt
│
├── requirements.txt
├── README.md
└── creditcard.csv                 # Dataset (download separately)
```

---

# 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** available publicly on Kaggle.

Dataset Characteristics:

- Highly imbalanced dataset
- Real-world anonymized transaction data
- PCA-transformed features (`V1`–`V28`)
- Additional features:
  - `Time`
  - `Amount`
- Target Variable:
  - **0 → Legitimate Transaction**
  - **1 → Fraudulent Transaction**

Since the dataset is relatively large, it is intentionally **excluded from GitHub** using `.gitignore`.

Download the dataset from Kaggle and place it in the project root as:

```
creditcard.csv
```

---

# ⚙️ Installation

Python **3.11** is recommended for TensorFlow compatibility.

Clone the repository:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git

cd credit-card-fraud-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser and allow you to test transaction fraud predictions interactively.

---

# 🧠 Model Training

Train the hybrid model using:

```bash
python train_and_save.py
```

The training pipeline performs the following steps:

1. Loads the dataset
2. Cleans and preprocesses the data
3. Trains an Autoencoder
4. Extracts compressed latent features
5. Generates contextual transaction features
6. Combines encoded and contextual features
7. Trains the Random Forest classifier
8. Saves all trained artifacts

Generated artifacts:

```
models/
│
├── encoder_model.h5
├── hybrid_scaler.joblib
├── rf_hybrid_model.joblib
└── feature_means.csv

reports/
└── training_report.txt
```

---

# 🔍 Prediction Pipeline

During inference:

1. User enters transaction details.
2. Input features are standardized.
3. The Autoencoder generates compressed feature representations.
4. Contextual transaction features are added.
5. The Random Forest classifier predicts the transaction class.
6. The application displays whether the transaction is likely **Fraudulent** or **Legitimate**.

---

# 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Random Forest
- Autoencoder Neural Network
- Pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn

---

# 📁 Saved Model Files

| File | Purpose |
|------|----------|
| encoder_model.h5 | Trained Autoencoder Encoder |
| hybrid_scaler.joblib | Feature scaler |
| rf_hybrid_model.joblib | Random Forest classifier |
| feature_means.csv | Mean values for preprocessing |

---

# 🌐 Deployment

The application can be deployed on:

- Streamlit Community Cloud
- Render
- Railway
- Hugging Face Spaces
- Azure App Service
- AWS EC2

---

# 📦 GitHub Large File Support (Git LFS)

Some model files exceed GitHub's **100 MB** file size limit.

Install Git LFS:

```bash
git lfs install
```

Track large model files:

```bash
git lfs track "models/*.joblib"
git lfs track "models/*.h5"
```

Commit the tracking configuration:

```bash
git add .gitattributes
git add models/
```

---

# 📈 Future Improvements

Some potential enhancements include:

- XGBoost and LightGBM integration
- Explainable AI (SHAP/LIME)
- Real-time fraud detection APIs
- REST API with FastAPI
- Docker containerization
- User authentication
- Transaction history dashboard
- Model monitoring and drift detection
- Hyperparameter optimization
- Cloud deployment with CI/CD pipelines

---

# ⚠️ Important Notes

- This project is intended for **educational, demonstration, and portfolio purposes**.
- The contextual transaction features used during training are **synthetically generated**, as they are not available in the original public dataset.
- For real-world deployment, the model should be retrained using actual:
  - Merchant information
  - Device fingerprints
  - Customer behavior
  - Geolocation data
  - IP address information
  - Historical transaction patterns

---

# 👨‍💻 Author

**Narayan Singh**

If you found this project useful, consider giving it a ⭐ on GitHub.
