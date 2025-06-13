# **DoclingQ&A**   

---

## **Abstract**  

**DoclingQ&A** is a **multi-step Retrieval-Augmented Generation (RAG) system** designed to help users query **long, complex documents** with **accurate, fact-verified answers**. Unlike traditional chatbots like **ChatGPT or DeepSeek**, which **hallucinate responses and struggle with structured data**, DoclingQ&A **retrieves, verifies, and corrects** answers before delivering them.  

 **Key Features:**  
✅ **Multi-Step System** – A **Research Agent** generates answers, while a **Verification Step** fact-checks responses.  
✅ **Hybrid Retrieval** – Uses **BM25 and vector search** to find the most relevant content.  
✅ **Handles Multiple Documents** – Selects the most relevant document even when multiple files are uploaded.  
✅ **Scope Detection** – Prevents hallucinations by **rejecting irrelevant queries**.  
✅ **Fact Verification** – Ensures responses are accurate before presenting them to the user.  
✅ **Web Interface with Gradio** – Allowing seamless document upload and question-answering.  

-

## ** How DoclingQ&A Works**  

### **1️⃣ Query Processing & Scope Analysis**  
- Users **upload documents** and **ask a question**.  
- DoclingQ&A **analyzes query relevance** and determines if the question is **within scope**.  
- If the query is **irrelevant**, DoclingQ&A **rejects it** instead of generating hallucinated responses.  

### **2️⃣ Multi-Agent Research & Retrieval**  
- **Docling** parses documents into a structured format (Markdown, JSON).  
- **LangChain & ChromaDB** handle **hybrid retrieval** (BM25 + vector embeddings).  
- Even when **multiple documents** are uploaded, **DoclingQ&A finds the most relevant sections** dynamically.  

### **3️⃣ Answer Generation & Verification**  
- **Research Agent** generates an answer using retrieved content.  
- **Verification Step** cross-checks the response against the source document.  
- If **verification fails**, a **self-correction loop** re-runs retrieval and research.  

### **4️⃣ Response Finalization**  
- **If the answer passes verification**, it is displayed to the user.  
- **If the question is out of scope**, DoclingQ&A informs the user instead of hallucinating.  

---

## **🎯 Why Use DoclingQ&A Instead of ChatGPT or DeepSeek?**  

| Feature | **ChatGPT/DeepSeek** ❌ | **DoclingQ&A** ✅ |
|---------|-----------------|---------|
| Retrieves from uploaded documents | ❌ No | ✅ Yes |
| Handles multiple documents | ❌ No | ✅ Yes |
| Extracts structured data from PDFs | ❌ No | ✅ Yes |
| Prevents hallucinations | ❌ No | ✅ Yes |
| Fact-checks answers | ❌ No | ✅ Yes |
| Detects out-of-scope queries | ❌ No | ✅ Yes |

🚀 **DoclingQ&A is built for enterprise-grade document intelligence, research, and compliance workflows.**  

---

## **📦 Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/mukundan1/docqa.git 
cd docqa
```

### **2️⃣ Set Up Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```bash
uv pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**  
Requires an OpenAI API key for processing. Add it to a `.env` file:
```bash
OPENAI_API_KEY=your-api-key-here
```

### **5️⃣ Run the Application** 
```bash
python app.py
```

DoclingQ&A will be accessible at `http://0.0.0.0:7860`.


## 🖥️ Instructions to use  

1️⃣ **Upload one or more documents** (PDF, JSON, DOCX, TXT, Markdown).  

2️⃣ **Enter a question** related to the document.  

3️⃣ **Click "Submit"** – DoclingQ&A retrieves, analyzes, and verifies the response.  

4️⃣ **Review the answer & verification report** for confidence.  

5️⃣ **If the question is out of scope**, DoclingQ&A will inform instead of hallucination.  

## Training Module with Guided Project - provided by IBM Skills Network
an 1-hour implementation.   