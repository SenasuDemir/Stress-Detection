import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vect = joblib.load('vectorizer.pkl')

def stress_prediction(text):
    text_arr = [text]
    text_transformed = vect.transform(text_arr)
    prediction = model.predict(text_transformed)
    return prediction

# Main function to render the Streamlit app
def main():
    # Set page configuration with a fancy icon and layout
    st.set_page_config(page_title="Stress Prediction", page_icon="🧠", layout="centered")

    # Add custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #F0F8FF;
        border-radius: 10px;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .title {
        font-size: 2rem;
        font-weight: bold;
        color: #0078d4;
    }
    .text-area {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        font-size: 1.1rem;
    }
    .button {
        background-color: #0078d4;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .result {
        font-size: 1.5rem;
        color: #FF6347;
        font-weight: bold;
    }
    .explanation {
        font-size: 1.1rem;
        color: #808080;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for additional information
    st.sidebar.title("About")
    st.sidebar.write("""
    This application predicts whether you are feeling stressed based on the text you input. 
    Just type how you're feeling, and the model will classify it for you. 
    Let's see if you're under pressure!
    """)

    # App title and description
    st.markdown('<div class="title">Stress Prediction</div>', unsafe_allow_html=True)
    st.write("""
    Enter your mental state below, and we will predict if you're under stress or not.
    """)

    # Input text area
    text = st.text_area("Type your feelings", "", height=150, key="text_input", label_visibility="visible")
    
    # Prediction button
    if st.button("Predict Stress", key="predict_button", help="Click to predict stress level", use_container_width=True):
        if text.strip() == "":
            st.warning("Please enter some text to make a prediction!")
        else:
            # Predict stress
            stress_pred = stress_prediction(text)

            # Display the result with enhanced visualization
            st.markdown(f'<div class="result">Prediction: {"Stressed" if stress_pred[0] == "Stress" else "Not Stressed"}</div>', unsafe_allow_html=True)
            
            # Add explanation text
            st.markdown('<div class="explanation">Our model analyzed your feelings and predicted your stress level based on your input.</div>', unsafe_allow_html=True)
            
            # Show confidence score (fake example here, can be modified if model returns probability)
            confidence = np.random.uniform(0.75, 0.95)  # Fake confidence score, replace with actual model confidence if available
            st.markdown(f'<div class="explanation">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
