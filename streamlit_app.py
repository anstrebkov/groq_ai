import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def main():

    st.title("GRSbot")

    # Добавление опций настройки на боковую панель
    st.sidebar.title('Выберите модель LLM')
    model = st.sidebar.selectbox(
        'Выберите модель',
        ['mixtral-8x7b-32768', "llama3-8b-8192",'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider(''Conversational memory length:', 1, 10, value = 5)

    memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Задайте свой вопрос:")

    # переменная состояния сессии
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input':message['human']},{'output':message['AI']})


    # Инициализация объекта ChatGroq и цепочки диалога
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )

    if user_question:
        response = conversation(user_question)
        message = {'human':user_question,'AI':response['response']}
        st.session_state.chat_history.append(message)
        st.write("GRSbot:", response['response'])

if __name__ == "__main__":
    main()

