from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document

URL = "https://delta-io.github.io/delta-rs/"
CHUNK_SIZE = 500


def get_docs() -> tuple[list[Document], RecursiveCharacterTextSplitter]:
    loader = WebBaseLoader(URL)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=20,
    )
    return docs, text_splitter


def get_split_docs() -> list[Document]:
    docs, text_splitter = get_docs()
    split_docs = text_splitter.split_documents(docs)

    return split_docs


def create_vector_store(docs: list[Document]) -> VectorStore:
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store
