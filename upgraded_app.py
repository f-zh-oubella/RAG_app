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

if 'current_model' not in st.session_state:
    st.session_state.current_model = "llama3.1"  

AVAILABLE_MODELS = {
    "Llama 3.1 (Default)": "llama3.1",
    "Llama 3": "llama3",
    "Mistral": "mistral"
}

# default parameters in session state
if 'params' not in st.session_state:
    st.session_state.params = {
        'chunk_size': 300,
        'chunk_overlap': 200,
        'k': 4,
        'fetch_k': 8,
        'score_threshold': 0.5
    }

@st.cache_resource
def initialize_models(model_name: str):
    return {
        'llm': Ollama(model=model_name, base_url="http://127.0.0.1:11434"),
        'embeddings': OllamaEmbeddings(model=model_name, base_url="http://127.0.0.1:11434")
    }

with st.sidebar:
    st.markdown("### Model Selection")
    
    selected_model = st.selectbox(
        "Choose LLM Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(st.session_state.current_model),
        help="Select the language model to use for processing"
    )
    model_name = AVAILABLE_MODELS[selected_model]
    if model_name != st.session_state.current_model:
        st.session_state.current_model = model_name
        st.session_state.vector_store = None  # Clear vector store when model changes
        st.experimental_rerun()

    st.markdown("### Parameter Configuration")
    
    st.markdown("#### Text Splitting Parameters")
    st.markdown("""
    These parameters control how the PDF document is split into chunks for processing:
    """)
    
    st.session_state.params['chunk_size'] = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=st.session_state.params['chunk_size'],
        step=50,
        help="Number of characters in each text chunk. Smaller chunks are better for specific information, larger chunks preserve more context."
    )

    st.session_state.params['chunk_overlap'] = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=st.session_state.params['chunk_size'],
        value=min(st.session_state.params['chunk_overlap'], st.session_state.params['chunk_size']),
        step=50,
        help="Number of overlapping characters between chunks. Higher overlap helps maintain context between chunks."
    )

    st.markdown("#### Retrieval Parameters")
    st.markdown("""
    These parameters control how the system searches for relevant information:
    """)
    
    st.session_state.params['k'] = st.slider(
        "Number of Chunks (k)",
        min_value=1,
        max_value=10,
        value=st.session_state.params['k'],
        help="Number of most relevant text chunks to retrieve. Higher values get more context but may include less relevant information."
    )

    st.session_state.params['fetch_k'] = st.slider(
        "Fetch Candidates (fetch_k)",
        min_value=st.session_state.params['k'],
        max_value=20,
        value=max(st.session_state.params['fetch_k'], st.session_state.params['k']),
        help="Number of candidates to consider before selecting the most relevant chunks. Should be >= k."
    )

    st.session_state.params['score_threshold'] = st.slider(
        "Relevance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.params['score_threshold'],
        step=0.05,
        help="Minimum similarity score for chunk selection. Lower values include more results but may be less relevant."
    )
    if st.button("Reset to Defaults"):
        st.session_state.params = {
            'chunk_size': 300,
            'chunk_overlap': 200,
            'k': 4,
            'fetch_k': 8,
            'score_threshold': 0.5
        }
        st.session_state.current_model = "llama3.1"
        st.session_state.vector_store = None
        st.experimental_rerun()

models = initialize_models(st.session_state.current_model)

st.title("PDF Chat with General Knowledge")
st.write(f"Currently using: {selected_model}")
st.write("Upload a PDF and ask questions about it or any general topic!")

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.params['chunk_size'],
        chunk_overlap=st.session_state.params['chunk_overlap'],
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    pdf_chunks = text_splitter.split_documents(pages)
    
    vector_store = Chroma.from_documents(
        documents=pdf_chunks,
        embedding=models['embeddings'],
        persist_directory="./pdf_chroma_db"
    )
    vector_store.persist()
    
    os.unlink(pdf_path)
    return vector_store

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

with col2:
    if st.session_state.vector_store is not None:
        if st.button("Clear PDF"):
            st.session_state.vector_store = None
            st.experimental_rerun()

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("PDF processed successfully!")

def get_response(question) -> Iterator[str]:
    try:
        if st.session_state.vector_store is not None:
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": st.session_state.params['k'],
                    "fetch_k": st.session_state.params['fetch_k'],
                    "score_threshold": st.session_state.params['score_threshold'],
                }
            )
            
            prompt = PromptTemplate.from_template("""You are a helpful AI assistant that provides accurate information from the given context. 
            Please analyze the following context carefully and answer the question.
            If the exact information is not found in the context, look for related or partial information that might help.
            If you cannot find any relevant information in the context, say so and provide a general response based on your knowledge.

            Context: {context}
            Question: {input}

            Answer: """)
            
            combine_docs_chain = create_stuff_documents_chain(
                llm=models['llm'],
                prompt=prompt
            )
            
            retrieval_chain = create_retrieval_chain(
                retriever,
                combine_docs_chain
            )
            
            response = retrieval_chain.invoke({
                "input": question
            })
            
            words = response['answer'].split()
            for word in words:
                yield word + " "
        else:
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
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.user_input = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])