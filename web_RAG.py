import streamlit as st
from RAG_app import load_split, RAG_app
from keys import LANGCHAIN_API_KEY, HUGGINGFACEHUB_API_TOKEN
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# App Configuration
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Ask your documents 💬")

DATA_FOLDER = "data"
CHROMA_PATH = "chroma"

# Ensure the data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Sidebar with context management
with st.sidebar:
    st.title("Upload your files below:")

    # File uploader in the sidebar
    uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        for uploaded_file in uploaded_files:
            #save to 'data' folder 
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


if "user_question" not in st.session_state:
    st.session_state.user_question = ""  # Initialize session state

if "chunks" not in st.session_state:
    st.session_state.chunks = None  # Initialize chunks state

if uploaded_files:
    if st.button("Process Files"):
        with st.spinner(":blue[*Processing your files...*]"):
            st.session_state.chunks = load_split(DATA_FOLDER)  # Store chunks in session state
        st.success("Done!🎉")

    if st.session_state.chunks:
        # Text input for the user's question
        user_question = st.text_input("**Write your question:**", value=st.session_state.user_question)
        st.session_state.user_question = user_question  # Update session state with the current input

        if user_question:
            with st.spinner("Generating response..."):
                response = RAG_app(user_question, st.session_state.chunks)
            st.text(response)
            st.session_state.user_question = ""
else:
    st.info("Please upload at least one file using the sidebar.")
