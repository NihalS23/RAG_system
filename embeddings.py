from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to generate embeddings for the contexts in the DataFrame
def generate_embeddings(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer(model_name)
    embeddings = []

    # Loop through each context in the DataFrame and generate embeddings
    for context in tqdm(df['context'].tolist(), desc="Generating Embeddings"):
        embedding = model.encode(context, convert_to_tensor=True)  # Generate embedding for the context
        embeddings.append(embedding)
    
    # Convert list of embeddings to a numpy array
    embeddings = np.vstack(embeddings)
    
    # Save the embeddings to a .npy file
    np.save('context_embeddings1.npy', embeddings)
    
    # Save the DataFrame with contexts to a CSV file
    df.to_csv('contexts_with_embeddings2.csv', index=False)

if __name__ == "__main__":
    # Load the preprocessed data from the CSV file
    df = pd.read_csv('processed_squad3.csv')
    
    # Generate embeddings for the contexts
    generate_embeddings(df)
