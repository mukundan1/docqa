import os
from langchain_community.document_loaders import TextLoader

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