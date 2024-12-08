import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import os

def extract_questions_and_answers(full_response):
    # """
    # Extract all questions and answers from a full response string and return them as separate arrays.
    # """
    # # Initialize arrays for questions and answers
    # questions = []
    # answers = []

    # # Split the full response by lines
    # lines = full_response.split("\n")

    # # Iterate over each line and categorize as question or answer
    # for line in lines:
    #     if line.startswith("Answer:"):
    #         answers.append(line.replace("Answer:", "").strip())
    #     if line.startswith("Question:"):
    #         questions.append(line.replace("Question:", "").strip())

    # return answers, questions

    """
    Extract all questions and answers from a full response string and return them as separate arrays.
    """
    # Initialize arrays for questions and answers
    questions = []
    answers = []

    # Split the full response by lines
    lines = full_response.split("\n")

    temp_answer = []
    temp_question = []

    isAnswer = False
    start = False

    # Iterate over each line and categorize as question or answer
    for line in lines:
        # Clean up the line by stripping extra spaces and replacing the labels
        line = line.strip()

        if line.startswith("Answer:"):
            # Add previous question to the list if any
            if temp_question:
                questions.append(" ".join(temp_question))
            temp_question.clear()
            start = True
            isAnswer = True
            line = line.replace("Answer:", "").strip()  # Remove "Answer:" label

        if line.startswith("Question:"):
            # Add previous answer to the list if any
            if temp_answer:
                answers.append(" ".join(temp_answer))
            temp_answer.clear()
            start = True
            isAnswer = False
            line = line.replace("Question:", "").strip()  # Remove "Question:" label

        # Depending on whether it's an answer or a question, append the line
        if start and isAnswer:
            temp_answer.append(line)
        else:
            temp_question.append(line)

    # Append the last answer and question if any remain
    if temp_answer:
        answers.append(" ".join(temp_answer))
    if temp_question:
        questions.append(" ".join(temp_question))

    # print("Answers: ")
    # for a in answers:
    #     print(f"<> <>. {a}")
    # print("Questions: ")
    # for  a in questions:
    #     print(f" <> <>. {a}")

    return answers, questions


# Helper function to resolve paths relative to the current script
def resolve_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


# Function to load CSV and preprocess with unique metadata
def load_and_preprocess_csv(csv_file_path, metadata_config):
    """
    Load a CSV file and preprocess it into LangChain Document format with unique metadata.
    """
    csv_file_path = resolve_path(csv_file_path)  # Ensure relative paths are resolved
    data = pd.read_csv(csv_file_path)

    # Ensure required columns exist
    assert "question" in data.columns and "answer" in data.columns, "CSV must contain 'question' and 'answer' columns."

    # Handle missing data
    data = data.dropna(subset=["answer"])  # Drop rows where 'answer' is NaN
    data["answer"] = data["answer"].fillna("No answer provided.")  # Fill NaN with placeholder if needed

    # Add metadata and convert to LangChain documents
    documents = [
        Document(
            page_content=row["answer"],
            metadata={key: row.get(value, f"Unknown {key.capitalize()}") for key, value in metadata_config.items()}
        )
        for _, row in data.iterrows()
    ]
    return documents


# Function to build vector store
def build_vector_store(documents, store_name="faiss_index"):
    """
    Build a FAISS vector store from the given documents.
    """
    # Ensure the directory exists
    store_name = resolve_path(store_name)  # Ensure path is resolved
    os.makedirs(store_name, exist_ok=True)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_name)
    return vector_store


# Function to load vector store
def load_vector_store(store_name="faiss_index", embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Load a FAISS vector store.
    """
    store_name = resolve_path(store_name)  # Ensure path is resolved
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    return FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)


# Function to initialize retriever
def initialize_retriever(store_name="faiss_index", embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Initialize the retriever from a vector store.
    """
    print(f"Loading vector store from: {store_name}")
    vector_store = load_vector_store(store_name, embeddings_model)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# Function to initialize the generator
def initialize_generator(model_id="meta-llama/Llama-3.1-8B"):
    """
    Initialize a text-generation model pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


# Query handler
def rag_query_handler(query, retriever, generator):
    """
    Handle a single RAG query and return only the generated answer as a string.
    """
    # Retrieve relevant documents using the retriever
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine document contents into context

    # Define the prompt to be sent to the generator
    prompt = (
        f"You are a smart and helpful assistant for XACBANK bank organization. "
        f"Provide a single, accurate answer to the user's question based on the context below. "
        f"If the context does not contain the information, respond with 'I'm not sure.' "
        f"Include a what3words on address questions"
        f"Context:\n{context}\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Generate a response using the LLM
    generated_response = generator.invoke(prompt)
    print("generated reponse: ", generated_response)  # Debugging: Print the raw generated response

    # Parse the generated response
    if isinstance(generated_response, list) and "generated_text" in generated_response[0]:
        full_response = generated_response[0]["generated_text"].strip()  # Extract response from list
    elif isinstance(generated_response, str):
        full_response = generated_response.strip()  # Handle direct string response
    else:
        return "Error: Unable to generate response."  # Return error if response format is unexpected

    answers, questions = extract_questions_and_answers(full_response)
    print(answers, questions)
    first_answer = answers[0] if answers else ""
    first_question = questions[1] if len(questions) >= 2 else ""

    return first_answer ,first_question





def test_build_and_load_vector_store():
    """
    Test the build_vector_store and load_vector_store functions with two CSV files.
    """
    # First CSV file and metadata configuration
    csv_file_1 = "../data/branch_khas_2.csv"
    metadata_config_1 = {
        "branch_name": "branch_name",
        "address": "address",
        "phone": "phone",
        "timetable": "timetable",
        "open_time": "open_time",
        "close_time": "close_time",
    }

    
    # Second CSV file and metadata configuration
    csv_file_2 = "../data/faq_khas_2.csv"
    metadata_config_2 = {
        "metadata": "metadata_column"  # Assumes the second CSV has a single metadata column named "metadata_column"
    }

    csv_file_3 = "../data/product_khas.csv"
    metadata_config_3 = {
    "metadata": "metadata_column",
    "product_name": "product_name",
    "link": "link",
    "advantages": "advantages",
    "terms": "terms",
    "conditions": "conditions",
    "benefits": "benefits",
    "requirements": "requirements",
    }

    
    # Load and preprocess documents from both CSV files
    documents_1 = load_and_preprocess_csv(csv_file_1, metadata_config_1)
    documents_2 = load_and_preprocess_csv(csv_file_2, metadata_config_2)  # Use metadata column
    documents_3 = load_and_preprocess_csv(csv_file_3, metadata_config_3)
    # Combine documents
    all_documents = documents_1 + documents_2 + documents_3
    
    # Build vector store
    store_name = "faiss_index"
    build_vector_store(all_documents, store_name)
    
    # Initialize retriever
    retriever = initialize_retriever(store_name)
    assert retriever is not None, "Failed to initialize retriever."
    print("Vector store build and load test with two CSV files passed.")


def test_rag_query_handler():
    # Initialize retriever and generator
    retriever = initialize_retriever("faiss_index")  # Updated to use the combined index
    generator = initialize_generator()

    # Test query from the first CSV
    query_1 = "What is the address of the Shine Darhan branch?"
    response_1 = rag_query_handler(query_1, retriever, generator)
    assert response_1, "No response generated for query 1."
    print(f"Query 1 Test Passed. Response: {response_1}")

    # Test query from the second CSV
    query_2 = "What is qpay?"  # Example query based on second CSV content
    response_2 = rag_query_handler(query_2, retriever, generator)
    assert response_2, "No response generated for query 2."
    print(f"Query 2 Test Passed. Response: {response_2}")

    # Test query from the third CSV
    query_3 = "How to make an online purchase?"  # Example query based on second CSV content
    response_3 = rag_query_handler(query_3, retriever, generator)
    assert response_3, "No response generated for query 3."
    print(f"Query 3 Test Passed. Response: {response_3}")



# test_rag_query_handler()