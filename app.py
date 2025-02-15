import streamlit as st
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gtts import gTTS
import os

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Hugging Face API details
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_API_TOKEN = "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"  # Replace with your Hugging Face API token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Function to call Hugging Face Inference API
def query_huggingface(payload):
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to generate response"}

# Function to preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Healthcare chatbot logic using Mistral-7B
def healthcare_chatbot(user_input):
    user_input = preprocess_input(user_input).lower()

    # Predefined responses for common queries
    if "sneeze" in user_input or "sneezing" in user_input:
        response = "Frequent sneezing may indicate allergies or a cold. Consult a doctor if symptoms persist."
    elif "symptom" in user_input:
        response = "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        response = "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        response = "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        # Use Mistral-7B-Instruct for generating responses
        payload = {"inputs": f"User: {user_input}\nHealthcare Assistant:"}
        chatbot_response = query_huggingface(payload)

        if "error" in chatbot_response:
            response = "Sorry, I couldn't process your request."
        else:
            response = chatbot_response[0]["generated_text"]

    return response

# Function to generate and save speech
def text_to_speech(text):
    audio_file = "response.mp3"

    # Remove previous audio file if it exists
    if os.path.exists(audio_file):
        os.remove(audio_file)

    # Generate new speech audio
    tts = gTTS(text=text, lang="en")
    tts.save(audio_file)

    return audio_file

# Streamlit Web App
def main():
    st.title("Healthcare Assistant Chatbot")

    # Initialize session state variables
    if "response_audio" not in st.session_state:
        st.session_state.response_audio = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    # User input field
    user_input = st.text_input("How can I assist you today?", "")

    # Submit button
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)

            # Generate speech file
            audio_file = text_to_speech(response)

            # Update session state with new audio
            st.session_state.response_audio = audio_file
            st.session_state.last_query = user_input  # Save last query

    # Display audio recording on UI (so user can replay it)
    if st.session_state.response_audio:
        st.audio(st.session_state.response_audio, format="audio/mp3")

if __name__ == "__main__":
    main()
