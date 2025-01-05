from keys import LANGCHAIN_API_KEY , HUGGINGFACEHUB_API_TOKEN
from langchain import HuggingFaceHub, LLMChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

import shutil
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


DATA_FOLDER = "data"
CHROMA_PATH ="chroma"

def load_split(docPath):
    loader = DirectoryLoader(docPath,glob="*.pdf",show_progress=True)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(doc)
    #print(f"Split {len(doc)} document(s) into {len(chunks)} chunks.")
    return chunks

def RAG_app(query_text,chunks):
    
    chunks = load_split(DATA_FOLDER)
    # Create a new DB from the documents.
    db = FAISS.from_documents(chunks, HuggingFaceEmbeddings())

    #Retrieve documents
    retrieved_docs = db.similarity_search(query_text)

    #Prepare context
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    #Define prompt template
    PROMPT_TEMPLATE = """
    Use the following retrieved context to answer the query:

    {context}

    Query: {query}
    Answer:
    """

    #Create the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    #Initialize the LLM
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"max_length": 64, "temperature": 0.5}
        )
    )

    #Run the LLM chain with context and query
    response_text = llm_chain.run({"context": context_text, "query": query_text})

    #Extract the most relvent source
    sources = retrieved_docs[0].metadata.get("source", "Unknown") if retrieved_docs else "No sources available"

    #Format the final response
    formatted_response = f"Response: {response_text} \n Sources: {sources}"
    return formatted_response
    
if __name__ == '__main__':
    chunks = load_split(DATA_FOLDER)
    response = RAG_app("what is the email of Charlie Brown?",chunks)
    print(response)