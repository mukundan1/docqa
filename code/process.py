import chromadb
import os
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# Set up our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Loading the publications
def load_research_publications(documents_path):
    """Load research publications from .txt files and return as list of strings"""
    
    # List to store all documents
    documents = []
    
    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Extract content as strings and return
    publications = []
    for doc in documents:
        publications.append(doc.page_content)
    
    return publications


#Chunk
def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data


#Search and retrieval
def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""
    
    # Convert question to vector
    query_vector = embeddings.embed_query(query)
    
    # Search for similar content
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return relevant_chunks
    

#Insertion into vector database   
def insert_publications(collection: chromadb.Collection, publications: list[str]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for publication in publications:
        chunked_publication = chunk_publication(publication)
        embeddings = embed_documents(chunked_publication)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication,
        )
        next_id += len(chunked_publication)


