# � AqariLens

**AqariLens** is an AI-powered real estate price prediction app designed to help users estimate property prices with high precision using advanced machine learning and semantic intelligence.

## Built Using

- Sentence Transformers (Arabic and English semantic understanding)
- Random Forest Regression
- Streamlit for a modern, interactive web app experience

## 🚀 Project Overview

- ✅ Predict property prices in Egypt with real-time inputs
- ✅ Semantic understanding of City, Location, Neighborhood
- ✅ Clean and modern UI with instant predictions
- ✅ Confidence interval for every prediction
- ✅ Powered by Machine Learning and NLP

## 📊 Model Details

- **Model**: Random Forest Regressor
- **Text Features**: Embedded using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Numerical Features**: Bedrooms, Bathrooms, Area (all log-transformed)
- **Validation R² Score**: `62%`

## 🛠 How to Run Locally

### Clone the repository:

```bash
git clone https://github.com/OldAlexhub/AqariLens.git
cd AqariLens
Install required libraries:
```

```bash
pip install -r requirements.txt
Run the app:
```

```bash
streamlit run app.py
Open your browser at:
http://localhost:8501
```

## 📂 Project Structure

**File** **Purpose**
**app.py** Main Streamlit application
**AqariLens_rf_model.pkl** Trained Random Forest model
**unique_cities.pkl** Preprocessed city options
**unique_locations.pkl** Preprocessed location options
**unique_neighborhoods.pkl** Preprocessed neighborhood options
**requirements.txt** Python dependencies

## 🧐 How It Works

1- User selects City, Location, Neighborhood from dropdowns (with search)

2- User enters Bedrooms, Bathrooms, Area

3- Text inputs are embedded via multilingual SentenceTransformer

4- Features processed (log1p transformations)

5- Random Forest predicts log(price), converted to final EGP value

6- App displays price estimate + confidence metrics

## 🌟 Why AqariLens?

- Arabic and English semantic support

- No hardcoded mappings: real understanding

- Scalable architecture for future expansion

- Designed for real-world real estate chaos

- Future-proof and lightweight

## 👨💼 Developed by

**Mohamed Gad**
Founder of **_OldAlexHub_**

## 🛡 License

This project is licensed for educational and practical deployment purposes.
For full commercial usage or partnerships, contact OldAlexHub.
