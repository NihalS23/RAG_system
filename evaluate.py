import json
from rag_chain import create_rag_system, run_query

# Function to load the test dataset
def load_dev_data(dev_path):
    with open(dev_path, 'r') as file:
        data = json.load(file)
    return data['data']

# Function to extract questions and their corresponding answers from the dataset
def extract_questions_and_answers(data):
    queries = []
    expected_answers = []
    
    # Loop through each article
    for article in data:
        # Loop through each paragraph
        for paragraph in article['paragraphs']:
            # Loop through each question-answer pair
            for qa in paragraph['qas']:
                queries.append(qa['question'])  # Collect the question
                if 'answers' in qa and qa['answers']:
                    # If there are answers, collect the first one
                    expected_answers.append(qa['answers'][0]['text'])
                else:
                    # If no answers are present, set it as None (unanswerable question)
                    expected_answers.append(None)
    
    return queries, expected_answers

# Function to evaluate the performance
def evaluate_performance(dev_path='D:/HiWi/Data/dev-v2.0.json'):
    # Load the test dataset
    data = load_dev_data(dev_path)
    
    # Extract the questions and expected answers
    queries, expected_answers = extract_questions_and_answers(data)
    
    # Create the rag system
    rag_chain = create_rag_system()
    
    correct = 0
    total = len(queries)
    
    # Loop through each query and its expected answer
    for query, expected_answer in zip(queries, expected_answers):
        # Run the query through the rag system
        response = run_query(query, rag_chain)
        
        # Print the query, expected answer, and the response from the system
        print(f"Query: {query}\nExpected Answer: {expected_answer}\nResponse: {response}\n")
        
        # Check if the expected answer is in the response (case-insensitive)
        if expected_answer and expected_answer.lower() in response.lower():
            correct += 1

    # Calculate and print the accuracy of the system
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_performance()
