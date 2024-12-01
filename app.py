import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import tempfile
from typing import Iterator

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def initialize_models():
    return {
        'llm': Ollama(model="llama3.1", base_url="http://127.0.0.1:11434"),
        'embeddings': OllamaEmbeddings(model="llama3.1", base_url="http://127.0.0.1:11434")
    }

models = initialize_models()

st.title("PDF Chat with General Knowledge")
st.write("Upload a PDF and ask questions about it or any general topic!")

def process_pdf(pdf_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    vector_store = Chroma.from_documents(
        documents=pdf_chunks,
        embedding=models['embeddings'],
        persist_directory="./pdf_chroma_db"
    )
    vector_store.persist()
    
    # clean up temporary file
    os.unlink(pdf_path)
    
    return vector_store

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("PDF processed successfully!")

# chat interface
def get_response(question) -> Iterator[str]:
    try:
        # if we have a vector store ->try to get context-specific answer first
        if st.session_state.vector_store is not None:
            retriever = st.session_state.vector_store.as_retriever()
            
            # prompt template
            prompt = PromptTemplate.from_template("""Answer the following question based on the provided context. 
            If the question cannot be answered from the context, provide a general response based on your knowledge.
            
            Context: {context}
            Question: {input}
            
            Answer: """)
            
            # chain 
            combine_docs_chain = create_stuff_documents_chain(
                llm=models['llm'],
                prompt=prompt
            )
            
            # retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever,
                combine_docs_chain
            )
            
            response = retrieval_chain.invoke({
                "input": question
            })
            
            # streaming the response word by word
            words = response['answer'].split()
            for word in words:
                yield word + " "
        else:
            # no PDF is uploaded -> use general knowledge
            response = models['llm'].invoke(question)
            words = response.split()
            for word in words:
                yield word + " "

    except Exception as e:
        yield f"An error occurred: {str(e)}"

st.text_input("Ask a question:", key="user_input", on_change=lambda: handle_user_input())

def handle_user_input():
    user_input = st.session_state.user_input
    if user_input:
        # user message 
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # clear input
        st.session_state.user_input = ""

# show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# usage instructions
with st.sidebar:
    st.markdown("""
    ### How to use:
    1. Upload a PDF document (optional)
    2. Ask questions about the PDF content
    3. Or ask any general knowledge questions
    
    The system will:
    - Use PDF content when relevant
    - Fall back to general knowledge for other questions
    """)