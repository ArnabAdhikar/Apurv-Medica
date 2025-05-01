# embedding-> Ctransformers
from langchain_huggingface import HuggingFaceEmbeddings
# vector stores/DB stores
from langchain_community.vectorstores import FAISS
# loading the document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# data paths
DATA_PATH = 'data/'
# store the embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Create vector database
def create_vector_db():
    # load the data(accept the pdf only)
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    # split this
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # pass this text to create the embeddings(HuggingFace)=> use CPU cores
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # create the vector database and store it
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
