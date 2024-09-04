import streamlit as st
import os
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Additional imports for PDF processing and langchain functionalities
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Setup device for language model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer for translation
language_model_name = "Qwen/Qwen2-1.5B-Instruct"
language_model = AutoModelForCausalLM.from_pretrained(
    language_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(language_model_name)

# Streamlit page configuration
st.set_page_config(
    page_title="Integrated AI Application",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Function to process input for translation
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

# Function to create vector DB from uploaded PDF
def create_vector_db(file_upload):
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())

    loader = UnstructuredPDFLoader(path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )

    shutil.rmtree(temp_dir)
    return vector_db

# Function to process user question using vector DB and selected model
def process_question(question, vector_db, selected_model):
    llm = ChatGroq(model=selected_model, api_key=groq_api_key)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = chain.invoke(question)
    return response

# Main function to run the Streamlit application
def main():
    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.selectbox("Select Mode", ["PDF RAG", "Translation", "Data Dashboard"])

    if app_mode == "PDF RAG":
        st.subheader("🧠 PDF Retrieval-Augmented Generation")

        models_info = {"models": [{"name": "model_1"}, {"name": "model_2"}]}  # Example, replace with real data
        available_models = [model["name"] for model in models_info["models"]]

        col1, col2 = st.columns([1.5, 2])

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = None

        if available_models:
            selected_model = col2.selectbox(
                "Pick a model available locally on your system ↓", available_models
            )

        file_upload = col1.file_uploader(
            "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
        )

        if file_upload:
            st.session_state["file_upload"] = file_upload
            if st.session_state["vector_db"] is None:
                st.session_state["vector_db"] = create_vector_db(file_upload)

            # Assuming extract_all_pages_as_images is defined somewhere or replaced with relevant code
            # pdf_pages = extract_all_pages_as_images(file_upload)
            # st.session_state["pdf_pages"] = pdf_pages

            # zoom_level = col1.slider(
            #     "Zoom Level", min_value=100, max_value=1000, value=700, step=50
            # )

            # with col1:
            #     for page_image in pdf_pages:
            #         st.image(page_image, width=zoom_level)

        delete_collection = col1.button("⚠️ Delete collection", type="secondary")

        if delete_collection:
            if st.session_state["vector_db"] is not None:
                st.session_state["vector_db"].delete_collection()
                st.session_state.pop("vector_db", None)
                st.success("Collection and temporary files deleted successfully.")
                st.rerun()
            else:
                st.error("No vector database found to delete.")

        with col2:
            message_container = st.container()

            for message in st.session_state["messages"]:
                avatar = "🤖" if message["role"] == "assistant" else "😎"
                with message_container.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Enter a prompt here..."):
                try:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    message_container.chat_message("user", avatar="😎").markdown(prompt)

                    with message_container.chat_message("assistant", avatar="🤖"):
                        with st.spinner(":green[processing...]"):
                            if st.session_state["vector_db"] is not None:
                                response = process_question(
                                    prompt, st.session_state["vector_db"], selected_model
                                )
                                st.markdown(response)
                            else:
                                st.warning("Please upload a PDF file first.")

                    if st.session_state["vector_db"] is not None:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )

                except Exception as e:
                    st.error(e, icon="⛔️")

    elif app_mode == "Translation":
        st.subheader("🌐 Translation and Chat")

        input_text = st.text_area("Input text")
        action = st.selectbox("Select action", ["Translate to English", "Translate to Chinese", "Translate to Japanese", "Translate to Russian", "Chat"])

        if st.button("Submit"):
            output_text, lang = process_input(input_text, action)
            st.text_area("Output text", value=output_text, height=200)
            audio_filename = text_to_speech(output_text, lang)
            audio_file = open(audio_filename, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')

    elif app_mode == "Data Dashboard":
        st.subheader("📊 Data Dashboard")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data")
            st.write(df)

            st.write("Data Visualization")

            st.bar_chart(df.select_dtypes(include=["float", "int"]))

            st.line_chart(df.select_dtypes(include=["float", "int"]))

if __name__ == "__main__":
    main()
