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
import chardet

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']


def main():
    st.title("GRSbot with Data Dashboard")

    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.selectbox("Select Mode", ["GRSbot", "Data Dashboard"])

    if app_mode == "GRSbot":
        st.subheader("GRSbot")

        st.sidebar.title('Select LLM Model')
        model = st.sidebar.selectbox(
            'Select Model',
            ['mixtral-8x7b-32768', 'llama2-70b-4096', "llama3-8b-8192", 'gemma-7b-it']
        )
        conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)

        memory = ConversationBufferWindowMemory(k=conversational_memory_length)

        user_question = st.text_area("Ask Your Question:")

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        for message in st.session_state['chat_history']:
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
            st.session_state['chat_history'].append(message)
            st.write("GRSbot:", response['response'])

    elif app_mode == "Data Dashboard":
        st.subheader("Simple Data Dashboard")

        uploaded_file = st.file_uploader("Upload CSV File", type="csv")

        if uploaded_file is not None:
            # Detect encoding
            rawdata = uploaded_file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            uploaded_file.seek(0)  # Reset file pointer after reading

            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                df.columns = df.columns.str.strip()  # Strip whitespace from column names
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
                return

            st.subheader("Data Preview")
            st.write(df.head())

            st.subheader("Data Summary")
            st.write(df.describe())

            st.subheader("Data Filtering")
            columns = df.columns.tolist()
            selected_column = st.selectbox("Select Column for Filtering", columns)
            unique_values = df[selected_column].unique()
            selected_value = st.selectbox("Select Value", unique_values)

            filtered_df = df[df[selected_column] == selected_value]
            st.write(filtered_df)

            st.subheader("Plotting Chart")
            x_column = st.selectbox("Select X-axis Column", columns)
            y_column = st.selectbox("Select Y-axis Column", columns)

            if st.button("Generate Chart"):
                if x_column in filtered_df.columns and y_column in filtered_df.columns:
                    try:
                        st.line_chart(filtered_df.set_index(x_column)[y_column])
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
                else:
                    st.error("Selected columns are not available in the filtered data.")

        else:
            st.write("Waiting for file upload...")


if __name__ == "__main__":
    main()
