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