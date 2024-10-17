import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Always run huggingface-cli login in terminal and enter you access token

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

def load_squad_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data['data']

def clean_text(text, lemmatizer, stop_words):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Rejoin words into cleaned text
    cleaned_text = ' '.join(words)
    
    return cleaned_text.strip()

def preprocess_squad(data):
    contexts = []
    questions = []
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    for article in data:
        for paragraph in article['paragraphs']:
            context = clean_text(paragraph['context'], lemmatizer, stop_words)
            for qa in paragraph['qas']:
                question = clean_text(qa['question'], lemmatizer, stop_words)
                contexts.append(context)
                questions.append(question)
    
    df = pd.DataFrame({'context': contexts, 'question': questions})

    # Remove duplicate rows
    df.drop_duplicates(subset=['context', 'question'], inplace=True)

    # Remove rows with empty or null context or question
    df.dropna(subset=['context', 'question'], inplace=True)
    df = df[df['context'].str.strip() != ""]
    df = df[df['question'].str.strip() != ""]
    
    return df

if __name__ == "__main__":
    dataset_path = 'D:/HiWi/Data/dev-v2.0.json'
    data = load_squad_data(dataset_path)
    df = preprocess_squad(data)
    df.to_csv('processed_squad3.csv', index=False)
