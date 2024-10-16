SQuAD 2.0 RAG-Chain System
Overview
This project implements a Retrieval-Augmented Generation (RAG) system using the SQuAD 2.0 dataset. The system retrieves relevant documents based on a user query and generates coherent responses by leveraging pre-trained language models and a vector database (FAISS) for efficient retrieval. The project uses LangChain (Community Edition) with the Chroma vector database as the storage backend, and the Llama2 model for response generation.

Key Features:
Loading and Preprocessing: Load and preprocess the SQuAD 2.0 dataset, including extracting context passages and questions.
Embedding Generation: Generate embeddings for the context passages using a pre-trained Sentence Transformer.
Document Indexing: Use Chroma vector database for efficient document indexing and retrieval.
RAG Chain System: Implement a Retrieval-Augmented Generation (RAG) system that retrieves relevant documents and generates responses.
Performance Evaluation: Evaluate the system using the SQuAD 2.0 development dataset (dev-v2.0.json) and calculate accuracy metrics.
Dependencies
The project requires the following Python packages, which are listed in the requirements.txt file:

transformers: Hugging Face's library for pre-trained models.
torch: PyTorch, a deep learning library used by many models in transformers.
sentence-transformers: A library for sentence embeddings using pre-trained models.
langchain-community: LangChain's community edition, used for creating the document index and retrieval pipeline.
llama3: Library for using the Llama3 language model.
chromadb: Chroma vector database for storing and retrieving document embeddings.
numpy: Fundamental package for numerical computations in Python.
pandas: Data analysis and manipulation library.
scikit-learn: Machine learning library for additional evaluation metrics.


Setup Guide
Step 1: Setup a Virtual Environment
- python3 -m venv venv
- .venv/Scripts/activate

Step 2: Install Dependencies
- pip install -r requirements.txt

Step 3: Login into Hugging Face CLI 
terminal - huggingface-cli login
Enter the access token


Running the Project
Step 1: data_processing.py
- This script will generate a processed_squad.csv file containing the context passages and questions.
- It also does basic cleaning, removes stop words, and does lemmatization.

Step 2: embeddings.py
- This will create a context_embeddings.npy file containing the embeddings.

Step 3: indexing.py
- The Chroma index will be stored locally for retrieval. Unfortunately, Chroma was not working for me so I used FAISS instead!!!

Step 4: rag-chain.py
- This will output a response based on the user query and the retrieved documents.

Step 5: evaluate.py
- This will print the accuracy of the system based on the retrieved documents and generated responses.
