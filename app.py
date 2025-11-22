import streamlit as st
import pickle

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

# Page configuration
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Mail Detection")
st.write("Enter an email message below to check if it's spam or not.")

# Load model
try:
    vectorizer, model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model files not found. Please ensure 'tfidf_vectorizer.pkl' and 'random_forest_model.pkl' are in the same directory.")

# Text input
message = st.text_area(
    "Email Message",
    height=150,
    placeholder="Type or paste your email message here..."
)

# Predict button
if st.button("Check for Spam", type="primary"):
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    elif model_loaded:
        # Transform the input and predict
        input_features = vectorizer.transform([message])
        prediction = model.predict(input_features)[0]
        
        # Display result
        if prediction == 1:
            st.success("✅ **Not Spam (Ham)**")
            st.write("This message appears to be legitimate.")
        else:
            st.error("🚨 **Spam Detected!**")
            st.write("This message appears to be spam.")
        
        # Show prediction probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_features)[0]
            st.write("---")
            col1, col2 = st.columns(2)
            col1.metric("Spam Probability", f"{proba[0]*100:.1f}%")
            col2.metric("Ham Probability", f"{proba[1]*100:.1f}%")

# Footer
st.write("---")
st.caption("Built with Streamlit | Model: Random Forest Classifier")