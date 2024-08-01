import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import torch
import io
import os

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\AI Major\\Major_Group_Project\\temp_fine-tuned-model")

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def text_to_speech(text, lang='hi'):
    tts = gTTS(text=text, lang=lang)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

def recognize_speech_and_translate():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
        english_text = recognizer.recognize_google(audio)
        
    hindi_text = translate_text(english_text, tokenizer, model)
    audio_file = text_to_speech(hindi_text)
    
    return english_text, hindi_text, audio_file
