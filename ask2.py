from transformers import pipeline
import chromadb

# Initialize the Hugging Face model
qa_pipeline = pipeline("question-answering")

CHROMA_PATH = r"chroma_db"
# Initialize the ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)


# Function to get answer from ChromaDB
def get_answer_from_chromadb(query):
    # Assuming you have a collection named 'qa_collection' in ChromaDB
    collection = client.get_collection("growing_vegetables")
    
    # Search for the query in the collection
    results = collection.search(query)
    
    if results:
        return results[0]['answer']
    else:
        return "No answer found in the database."

# Function to get answer from Hugging Face model
def get_answer_from_model(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Main function to handle user query
def handle_user_query(query):
    # First, try to get the answer from ChromaDB
    answer = get_answer_from_chromadb(query)
    
    if answer == "No answer found in the database.":
        # If no answer found in ChromaDB, use the Hugging Face model
        context = "Provide some context here if available."
        answer = get_answer_from_model(query, context)
    
    return answer

# Example usage
if __name__ == "__main__":
    user_query = "What vegetables grown in Florida?"
    answer = handle_user_query(user_query)
    print(f"Answer: {answer}")