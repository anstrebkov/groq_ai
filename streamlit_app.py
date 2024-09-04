import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS

# Setup device to always use CPU
device = "cpu"

# Load a known and supported model
language_model_name = "gpt2"  # Replace with a valid model name
language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

def process_input(input_text, action):
    if action == "Translate to English":
        prompt = f"Please translate the following text into English: {input_text}"
        lang = "en"
    elif action == "Translate to Chinese":
        prompt = f"Please translate the following text into Chinese: {input_text}"
        lang = "zh-cn"
    elif action == "Translate to Japanese":
        prompt = f"Please translate the following text into Japanese: {input_text}"
        lang = "ja"
    elif action == "Translate to Russian":
        prompt = f"Please translate the following text into Russian: {input_text}"
        lang = "ru"
    else:
        prompt = input_text
        lang = "en"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = language_model.generate(inputs['input_ids'], max_new_tokens=512)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text, lang

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    filename = "output_audio.mp3"
    tts.save(filename)
    return filename

# Streamlit app interface
st.title("Translation and Chat App")
input_text = st.text_area("Input text")
action = st.selectbox("Select action", ["Translate to English", "Translate to Chinese", "Translate to Japanese", "Translate to Russian"])

if st.button("Submit"):
    output_text, lang = process_input(input_text, action)
    st.text_area("Output text", value=output_text, height=200)

    audio_filename = text_to_speech(output_text, lang)
    with open(audio_filename, 'rb') as audio_file:
        st.audio(audio_file, format="audio/mp3")
