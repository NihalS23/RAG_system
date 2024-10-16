from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Function to generate and save a FAISS index from embeddings
def generate_faiss_index(embedding_file='context_embeddings1.npy'):
    embeddings = np.load(embedding_file)  # Load the embeddings
    d = embeddings.shape[1]  # Dimension of the embeddings
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(d)
    
    # Add the embeddings to the index
    index.add(embeddings)
    
    # Save the index to a file
    faiss.write_index(index, 'faiss_index.index')
    print(f"Index with {index.ntotal} embeddings created")

# Function to search the index with a query embedding
def search_faiss(query_embedding, index_file='faiss_index.index'):
    # Load the index from the file
    index = faiss.read_index(index_file)
    
    # Perform a search in the index to find the top 5 closest embeddings
    D, I = index.search(query_embedding, k=5)
    
    return I

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('processed_squad3.csv')
    
    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Define a query for testing and generate its embedding
    query = "What is SQuAD dataset?"
    query_embedding = model.encode([query])
    
    # Generate and save the index
    generate_faiss_index()
    
    # Search the FAISS index with the query embedding
    results = search_faiss(query_embedding)
    
    # Print the indices
    print("Top 5 results indices:", results)
