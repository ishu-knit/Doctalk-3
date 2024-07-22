import streamlit as st
from PyPDF2 import PdfReader
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




# loading css 
with open(".streamlit/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>' , unsafe_allow_html=True)


def main():
    st.title("Doctalk-3")
    st_lottie(lottie_animation, key="hello", height=100)

    # File upload widget
    doc_reader = st.file_uploader("Upload a PDF file", type=["pdf"])

    if doc_reader and "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            # Text extraction
            raw_text = ''
            for i, page in enumerate(PdfReader(doc_reader).pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ".",
                    ",",
                    "\u200b",  # Zero-width space
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ],
                chunk_size=500,
                chunk_overlap=10,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)

            #: Create embeddings and store in database
            try:
                ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url='http://127.0.0.1:11434')
                model = Ollama(model="llama3", base_url='http://127.0.0.1:11434')


                vectorstore = FAISS.from_texts(texts, ollama_embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.ollama_embeddings = ollama_embeddings
                st.session_state.model = model
                st.success("Text processed and embeddings created!")

            except Exception as e:
                st.warning(f"Failed to initialize ollama embeddings: {str(e)}")
                st.stop()

    if "vectorstore" in st.session_state:
        # retrieval F(x)
        if "retrieval_chain" not in st.session_state:

            try:
                
                template = """
                Use the following context and 
                ------
                <context>{context}</context>
                ------
                ------
                    {question}
                Answer: and also give the answer from the pdfs not from other sources
                """



                prompt = PromptTemplate(
                    input_variables=[ "context", "question"],
                    template=template,
                )

                Retrieval_fx = st.session_state.vectorstore.as_retriever()
                chain = RetrievalQA.from_chain_type(
                    st.session_state.model,
                    retriever=Retrieval_fx,
                    chain_type_kwargs={"prompt": prompt}
                )
                st.session_state.retrieval_chain = chain
                st.success("Retrieval chain created!")
            except Exception as e:
                st.warning(f"Failed to load question answering chain: {str(e)}")
                st.stop()

        # User query input and processing
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

main()
