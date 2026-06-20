import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'  # Folder to store PDFs


def upload_pdf(file):
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)

    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# Step 2: Create Chunks
def create_chunks(documents, file_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)

    for chunk in text_chunks:
        chunk.metadata["source"] = file_name

    return text_chunks


# Step 3: Setup Embeddings Model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


embedding_model = get_embedding_model()
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output size

# Step 4: Setup Pinecone Client + Index
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ai-lawyer")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

vector_store = PineconeVectorStore(index=pinecone_index, embedding=embedding_model)


# Step 5: Index Documents with Metadata
def index_pdf(file_path):
    documents = load_pdf(file_path)
    file_name = file_path.split("/")[-1]
    text_chunks = create_chunks(documents, file_name)

    vector_store.add_documents(text_chunks)
    return vector_store


# Step 6: Retrieve Docs with Filtering (server-side metadata filter)
def retrieve_docs(query, file_name):
    retrieved_docs = vector_store.similarity_search(
        query,
        k=5,
        filter={"source": file_name},
    )
    return retrieved_docs