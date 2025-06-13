from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""
    
    # Get relevant research chunks
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    
    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}" 
        for chunk in relevant_chunks
    ])
    
    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    
    return response.content, relevant_chunks

# Initialize LLM and get answer
llm = ChatGroq(model="llama3-8b-8192")
answer, sources = answer_research_question(
    "What are effective techniques for handling class imbalance?",
    collection, 
    embeddings, 
    llm
)

print("AI Answer:", answer)
print("\nBased on sources:")
for source in sources:
    print(f"- {source['title']}")