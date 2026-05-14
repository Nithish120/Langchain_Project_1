import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting....")
    loader=TextLoader("D:/Apps/Cursor/Projects/Repository/RAG/blog.txt",encoding="UTF-8")
    document=loader.load()
    print("Splitting")
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    text=text_splitter.split_documents(document)
    print(f"Created {len(text)} chunnks")
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("Ingesting")
    PineconeVectorStore.from_documents(text, embeddings, index_name=os.environ["INDEX_NAME"])
    print("finish")



  