import os


import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


from langchain_community.llms import HuggingFacePipeline

# --- Заголовок интерфейса ---
st.set_page_config(page_title="RetrievalQA Chat", layout="centered")
st.title("🤖 RetrievalQA ассистент")

# --- Кеширование загрузки моделей ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_vector_store(_embeddings):
    return FAISS.load_local('vector_store/tech_big', _embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen1.5-0.5B"  # локальная causal модель
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        temperature=0.2,
        do_sample=True,
        top_p=0.95,
        top_k=30,
        repetition_penalty=1.1,
        
    )
    return HuggingFacePipeline(pipeline=pipe)    

# --- Инициализация компонентов ---
embeddings = load_embeddings()
db = load_vector_store(embeddings)
llm = load_llm()
retriever = db.as_retriever()

# --- Prompt и RetrievalQA chain ---
prompt = PromptTemplate.from_template(
    "Ты — ассистент в образовании. Отвечай строго на русском языке.\n"
    "Если нет ответа на вопрос, то так и пиши: Я не знаю.\n\n"
    "Вот полезная информация для ответа:\n{context}\n\n"
    "Вопрос:\n{question}\n\n"
    "Развёрнутый ответ:"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# --- Интерфейс пользователя ---
user_question = st.text_input("Задайте  вопрос :", "Что такое большие данные?")

if st.button("Получить ответ") and user_question:
    with st.spinner("Генерация ответа..."):
        result = qa_chain(user_question)
        st.subheader("Ответ:")
        st.write( result.get('result').split('Развёрнутый ответ:')[1].strip())

        if result.get("source_documents"):
            st.subheader("Использованные источники:")
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Источник {i}:**\n {doc.page_content[:500]}")
