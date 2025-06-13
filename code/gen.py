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