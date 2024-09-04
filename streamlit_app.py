import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq  # Updated import
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import chardet

# Load environment variables for Groq API
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Streamlit page configuration
st.set_page_config(
    page_title="Integrated PDF RAG, GRSbot, and Data Dashboard",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Function to extract model names for Groq
@st.cache_resource(show_spinner=True)
def extract_model_names(models_info):
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

# Function to create vector DB from uploaded PDF
def create_vector_db(file_upload):
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

# Function to process user question using vector DB and selected model
def process_question(question, vector_db, selected_model):
    logger.info(f"Processing question: {question} using model: {selected_model}")
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
    logger.info("Question processed and response generated")
    return response

# Function to delete the vector database
def delete_vector_db(vector_db):
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

# Main function to run the Streamlit application
def main():
    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.selectbox("Select Mode", ["PDF RAG", "GRSbot", "Data Dashboard"])

    if app_mode == "PDF RAG":
        st.subheader("🧠 Groq PDF RAG playground")

        models_info = {"models": [{"name": "model_1"}, {"name": "model_2"}]}  # Example, replace with real data
        available_models = extract_model_names(models_info)

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
            pdf_pages = extract_all_pages_as_images(file_upload)
            st.session_state["pdf_pages"] = pdf_pages

            zoom_level = col1.slider(
                "Zoom Level", min_value=100, max_value=1000, value=700, step=50
            )

            with col1:
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

        delete_collection = col1.button("⚠️ Delete collection", type="secondary")

        if delete_collection:
            delete_vector_db(st.session_state["vector_db"])

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
                    logger.error(f"Error processing prompt: {e}")
            else:
                if st.session_state["vector_db"] is None:
                    st.warning("Upload a PDF file to begin chat...")

    elif app_mode == "GRSbot":
        st.subheader("GRSbot")

        st.sidebar.title('Select LLM Model')
        model = st.sidebar.selectbox(
            'Select Model',
            ['mixtral-8x7b-32768', 'llama2-70b-4096', "llama3-8b-8192", 'gemma-7b-it']
        )
        conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)

        memory = ConversationBufferWindowMemory(k=conversational_memory_length)

        user_question = st.text_input("Ask me anything:")

        if user_question:
            llm = ChatGroq(model=model, api_key=groq_api_key)
            conversation = ConversationChain(llm=llm, memory=memory)
            answer = conversation.run(user_question)

            st.markdown(f"**User**: {user_question}")
            st.markdown(f"**Bot**: {answer}")

            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            st.session_state["chat_history"].append((user_question, answer))

            st.write("Chat History:")
            for question, answer in st.session_state['chat_history']:
                st.markdown(f"**User**: {question}")
                st.markdown(f"**Bot**: {answer}")
        else:
            st.write("No chat history yet.")

    elif app_mode == "Data Dashboard":
        st.subheader("Data Dashboard")

        file_upload = st.file_uploader("Upload your CSV file", type=["csv"])

        if file_upload:
            data = pd.read_csv(file_upload)

            st.dataframe(data.head())

            if st.checkbox("Show data summary"):
                st.write(data.describe())

            columns = data.columns.tolist()
            x_axis = st.selectbox("Select X-axis", columns)
            y_axis = st.selectbox("Select Y-axis", columns)

            st.subheader(f"Plotting {y_axis} vs {x_axis}")
            plt.figure(figsize=(10, 5))
            plt.plot(data[x_axis], data[y_axis], marker='o')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f'{y_axis} vs {x_axis}')
            st.pyplot(plt)

if __name__ == "__main__":
    main()
