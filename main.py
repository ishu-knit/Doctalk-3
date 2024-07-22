import streamlit as st
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS    
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_lottie import st_lottie, st_lottie_spinner
import requests

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://lottie.host/c37665ee-a522-477c-8b38-ab6151e3d55e/LoF0g7cBLK.json")
lottie_download = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_t26law.json")
lottie_loader = load_lottieurl("https://lottie.host/3654e8d9-2cd8-4c88-a7c1-ae91890edd68/WVIi8BhZtr.json")

# Load CSS
with open(".streamlit/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main application
def main():
    st.title("Doctalk-3")
    st_lottie(lottie_animation, key="hello", height=100)
    
    doc_reader = st.file_uploader("Upload a PDF file", type=["pdf"])

    if doc_reader and "vectorstore" not in st.session_state:
        process_pdf(doc_reader)

    if "vectorstore" in st.session_state:
        setup_retrieval_chain()
        handle_query()

def process_pdf(doc_reader):
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(doc_reader)
        texts = split_text(raw_text)

        try:
            create_embeddings(texts)
            st.success("Text processed and embeddings created!")
        except Exception as e:
            st.warning(f"Failed to initialize Ollama embeddings: {str(e)}")
            st.stop()

def extract_text_from_pdf(doc_reader):
    raw_text = ''
    for i, page in enumerate(PdfReader(doc_reader).pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def split_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
        ],
        chunk_size=500,
        chunk_overlap=10,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def create_embeddings(texts):
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url='http://127.0.0.1:11434')
    model = Ollama(model="llama3", base_url='http://127.0.0.1:11434')
    vectorstore = FAISS.from_texts(texts, ollama_embeddings)
    # vectorstore = FAISS.load_local("doctalk_database", ollama_embeddings)
    st.session_state.vectorstore = vectorstore
    st.session_state.ollama_embeddings = ollama_embeddings
    st.session_state.model = model

def setup_retrieval_chain():
    if "retrieval_chain" not in st.session_state:
        try:
            prompt = PromptTemplate(
                input_variables=["context", "history" ,"question"],
                template="""
                Use the following context and 
                ------
                <context>{context}</context>
                ------
                <history>{history}</history>
                ------
                    {question}
                Answer:  give the answer according to pdf only and donot include extra information from outside pdf but give detail 
                ans and also include points if possible

                """
            )

            Retrieval_fx = st.session_state.vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                st.session_state.model,
                retriever=Retrieval_fx,
                chain_type_kwargs={
                    "prompt": prompt, 
                    "memory": ConversationBufferMemory(
                                memory_key="history",
                                input_key="question"),
                                    }
            )

            st.session_state.retrieval_chain = chain
            st.success("Retrieval chain created!")
        except Exception as e:
            st.warning(f"Failed to load question answering chain: {str(e)}")
            st.stop()

def handle_query():
    query = st.text_input('Type your query here...', key="query_input")
    if st.button('Submit'):
        if query:
            try:
                with st_lottie_spinner(lottie_loader, key="download", height=100):
                    answer = st.session_state.retrieval_chain.invoke({"query": query})
                    
                    st.markdown(answer["result"])
                st.success("Query processed")
            except Exception as e:
                st.warning(f"Failed to perform search and question answering: {str(e)}")

if __name__ == "__main__":
    main()
