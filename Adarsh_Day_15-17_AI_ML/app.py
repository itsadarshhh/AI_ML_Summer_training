import streamlit as st
import time
from audio_processor import recognize_speech_and_translate, translate_text, text_to_speech
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")

st.title("Real-time Language Translator")

# Function to handle real-time translation
def real_time_translation():
    st.write("Listening for audio...")
    english_text, hindi_text, audio_file = recognize_speech_and_translate()
    
    st.write(f"Recognized Text (English): {english_text}")
    st.write(f"Translated Text (Hindi): {hindi_text}")
    st.audio(audio_file.getvalue(), format="audio/mp3")

if st.button("Start Listening"):
    real_time_translation()

# Text input for manual translation
st.subheader("Manual Text Translation")
input_text = st.text_input("Enter text in English:")
if st.button("Translate"):
    if input_text:
        hindi_text = translate_text(input_text, tokenizer, model)
        st.write(f"Translated Text (Hindi): {hindi_text}")
        
        audio_file = text_to_speech(hindi_text)
        st.audio(audio_file.getvalue(), format="audio/mp3")
    else:
        st.error("Please enter some text to translate.")
