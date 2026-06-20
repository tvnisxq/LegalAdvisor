import os
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""


pdfs_directory = 'pdfs/'
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output size

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ai-lawyer")


def upload_pdf(file):
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def create_vector_store(text_chunks):
    embeddings = get_embedding_model()
    pinecone_index = get_pinecone_index()
    vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
    vector_store.add_documents(text_chunks)
    return vector_store


def retrieve_docs(vector_store, query):
    return vector_store.similarity_search(query, k=5)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file and user_query:
        upload_pdf(uploaded_file)
        documents = load_pdf(pdfs_directory + uploaded_file.name)
        text_chunks = create_chunks(documents)
        vector_store = create_vector_store(text_chunks)

        retrieved_docs = retrieve_docs(vector_store, user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("user").write(user_query)
        st.chat_message("AI Lawyer").write(response)

    else:
        st.error("Kindly upload a valid PDF file and/or ask a valid Question!")