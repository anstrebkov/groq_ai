import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
language_model_name = "Qwen/Qwen2-1.5B-Instruct"
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

# Function to process input
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

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = language_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text, lang

# Function to convert text to speech
def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    filename = "output_audio.mp3"
    tts.save(filename)
    return filename

# Streamlit app interface
st.title("Translation and Chat App using AI")
st.write("Translate input text or chat based on the selected action.")

# User inputs
input_text = st.text_area("Input text")
action = st.selectbox("Select action", ["Translate to English", "Translate to Chinese", "Translate to Japanese", "Translate to Russian", "Chat"])

if st.button("Submit"):
    # Process input and generate output
    output_text, lang = process_input(input_text, action)

    # Display the output text
    st.text_area("Output text", value=output_text, height=200)

    # Generate and play the audio
    audio_filename = text_to_speech(output_text, lang)
    audio_file = open(audio_filename, 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3")

