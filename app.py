import os
from dotenv import load_dotenv
from langchain.document_loaders import DoclingLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph import StateGraph, AgentState
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import List, Dict
import gradio as gr

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.cache_dir = "document_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process(self, files: List[str]) -> List[str]:
        chunks = []
        for file in files:
            # Load and process document
            loader = DoclingLoader(file)
            documents = loader.load()
            
            # Split into chunks using headers
            text_splitter = MarkdownHeaderTextSplitter(headers=self.headers_to_split_on)
            chunked_docs = text_splitter.split_text(documents[0].page_content)
            
            chunks.extend(chunked_docs)
        
        return chunks

class ResearchAgent:
    def __init__(self):
        credentials = Credentials(
            url=os.getenv("WATSONX_URL"),
        )
        self.model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 300,
                "temperature": 0.3,
            }
        )

    def generate(self, question: str, documents: List[str]) -> Dict:
        context = "\n\n".join(documents)
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.
        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}
        **Provide your answer below:**
        """
        
        response = self.model.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return {
            "draft_answer": response['choices'][0]['message']['content'],
            "context_used": context
        }

class VerificationAgent:
    def __init__(self):
        credentials = Credentials(
            url=os.getenv("WATSONX_URL"),
        )
        self.model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 200,
                "temperature": 0.0,
            }
        )

    def check(self, answer: str, documents: List[str]) -> Dict:
        context = "\n\n".join(documents)
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.
        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.
        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]
        **Answer:** {answer}
        **Context:**
        {context}
        **Respond ONLY with the above format.**
        """
        
        response = self.model.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response['choices'][0]['message']['content']

def process_documents(files):
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process(files)
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_texts(
        chunks,
        embeddings,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    )
    
    return docsearch

def get_answer(question, docsearch):
    # Retrieve relevant documents
    docs = docsearch.similarity_search(question, k=3)
    
    # Research phase
    research_agent = ResearchAgent()
    research_result = research_agent.generate(question, docs)
    
    # Verification phase
    verification_agent = VerificationAgent()
    verification_result = verification_agent.check(
        research_result["draft_answer"],
        docs
    )
    
    return {
        "answer": research_result["draft_answer"],
        "verification": verification_result,
        "context": research_result["context_used"]
    }

def chat_interface(question, files):
    if not files:
        return "Please upload documents first."
    
    docsearch = process_documents(files)
    result = get_answer(question, docsearch)
    
    return {
        "answer": result["answer"],
        "verification": result["verification"],
        "context": result["context"]
    }

# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# DCChat - Document Chat Application")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Documents",
                    file_types=[".pdf", ".docx", ".txt"],
                    multiple=True
                )
                
            with gr.Column():
                question_input = gr.Textbox(
                    label="Ask a question about the documents",
                    placeholder="Enter your question here..."
                )
                
        submit_btn = gr.Button("Submit")
        
        output = gr.JSON(label="Response")
        
        submit_btn.click(
            fn=chat_interface,
            inputs=[question_input, file_input],
            outputs=output
        )
    
    return demo

def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        with open(file.name, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)

if __name__ == "__main__":
    main()
