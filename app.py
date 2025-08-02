from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

# Set Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# LangChain tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demo with Google Gemini")
input_text = st.text_input("Search the topic you want")

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Show result
if input_text:
    st.write(chain.invoke({"question": input_text}))