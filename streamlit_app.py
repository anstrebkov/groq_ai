import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def main():
    st.title("GRSbot with Data Dashboard")

    # Sidebar for selecting the application mode
    st.sidebar.title("Выберите режим")
    app_mode = st.sidebar.selectbox("Выберите режим работы", ["GRSbot", "Data Dashboard"])

    if app_mode == "GRSbot":
        # GRSbot section
        st.subheader("GRSbot")

        st.sidebar.title('Выберите модель LLM')
        model = st.sidebar.selectbox(
            'Выберите модель',
            ['mixtral-8x7b-32768', 'llama2-70b-4096', "llama3-8b-8192",'gemma-7b-it']
        )
        conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

        memory = ConversationBufferWindowMemory(k=conversational_memory_length)

        user_question = st.text_area("Задайте свой вопрос:")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        else:
            for message in st.session_state.chat_history:
                memory.save_context({'input': message['human']}, {'output': message['AI']})

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
            message = {'human': user_question, 'AI': response['response']}
            st.session_state.chat_history.append(message)
            st.write("GRSbot:", response['response'])

    elif app_mode == "Data Dashboard":
        # Data Dashboard section
        st.subheader("Simple Data Dashboard")

        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')

            st.subheader("Предпросмотр данных")
            st.write(df.head())

            st.subheader("Сводка данных")
            st.write(df.describe())

            st.subheader("Фильтрация данных")
            columns = df.columns.tolist()
            selected_column = st.selectbox("Выберите колонку для фильтрации", columns)
            unique_values = df[selected_column].unique()
            selected_value = st.selectbox("Выберите значение", unique_values)

            filtered_df = df[df[selected_column] == selected_value]
            st.write(filtered_df)

            st.subheader("Построение графика")
            x_column = st.selectbox("Выберите колонку для оси X", columns)
            y_column = st.selectbox("Выберите колонку для оси Y", columns)

            if st.button("Сгенерировать график"):
                st.line_chart(filtered_df.set_index(x_column)[y_column])
        else:
            st.write("Ожидание загрузки файла...")

if __name__ == "__main__":
    main()

