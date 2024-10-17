import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Function to load the faiss index from a file
def load_faiss_index(index_file='faiss_index.index'):
    return faiss.read_index(index_file)

# Function to perform a search on the faiss index
def search_faiss(index, query_embedding, k=5):
    D, I = index.search(query_embedding, k)  # Get the top-k results
    return I

# Function to generate a response using the llama model
def generate_response_with_llama2(contexts, query, model, tokenizer, device):
    # Create a prompt combining the contexts and the query
    prompt = f"Answer the question based on the following contexts:\n\n" \
             f"{' '.join(contexts)}\n\nQuestion: {query}\nAnswer:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the response without updating model weights
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    
    # Decode the generated response and return it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Function to create the RAG system
def create_rag_system():
    # Load the faiss index
    index = load_faiss_index()

    # Load the Sentence Transformer model to generate query embeddings
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load the contexts data from a CSV file
    df = pd.read_csv('processed_squad3.csv')

    # Load the LLaMA2 model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    access_token = "Enter_your_token_here"  # Access token for authentication

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    response_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=access_token).to(device)

    return {
        "index": index,
        "embedding_model": embedding_model,
        "df": df,
        "response_model": response_model,
        "tokenizer": tokenizer,
        "device": device
    }

# Function to run a query using the RAG system
def run_query(query, rag_system, k=5):
    # Generate the query embedding
    query_embedding = rag_system['embedding_model'].encode([query])
    
    # Perform faiss search to retrieve relevant contexts
    result_indices = search_faiss(rag_system['index'], query_embedding, k=k)

    # Get the contexts corresponding to the search results
    retrieved_contexts = rag_system['df']['context'].iloc[result_indices[0]].tolist()

    # Generate a response using the retrieved contexts and the llama model
    response = generate_response_with_llama2(
        retrieved_contexts, query, 
        rag_system['response_model'], 
        rag_system['tokenizer'], 
        rag_system['device']
    )

    return response

if __name__ == "__main__":
    # Example usage of the RAG system
    rag_system = create_rag_system()
    query = "What is the purpose of the SQuAD dataset?"
    response = run_query(query, rag_system)
    print(f"Query: {query}\nResponse: {response}")
