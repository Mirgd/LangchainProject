import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from keys import LANGCHAIN_API_KEY, HUGGINGFACEHUB_API_TOKEN
from langchain import HuggingFaceHub, LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import os

# Environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# App Configuration
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chat with your documents")

DATA_FOLDER = "data"
CHROMA_PATH = "chroma"

# Ensure the data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Document Loader and Splitter
def load_split(doc_path):
    loader = DirectoryLoader(doc_path, glob="*.pdf", show_progress=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)

# RAG-based Application
def RAG_app(query_text):
    # Load and split documents
    chunks = load_split(DATA_FOLDER)

    # Clear out the existing Chroma database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new database from the documents
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()

    # Retrieve relevant documents
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query_text)

    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # Define prompt template
    PROMPT_TEMPLATE = """
    Use the following retrieved context to answer the query:

    {context}

    Query: {query}
    Answer:
    """

    # Initialize the LLM chain
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"max_length": 64, "temperature": 0.5}
        )
    )

    # Run the LLM chain
    response_text = llm_chain.run({"context": context_text, "query": query_text})
    sources = retrieved_docs[0].metadata.get("source", "Unknown") if retrieved_docs else "No sources available"

    # Format the final response
    return f"{response_text}\n\nSources: {sources}"

# Chat Helper
def get_response(user_query, chat_history):
    context = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in chat_history])
    return RAG_app(user_query)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# File Upload
uploaded_file = st.file_uploader("Upload a document (PDF only):", type=["pdf"])
if uploaded_file:
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# User Input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("AI"):
        st.write(response)
