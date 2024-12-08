from nltk.translate.bleu_score import sentence_bleu
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import os
from rag import initialize_generator, initialize_retriever
import pandas as pd
from google.cloud import translate_v2 as translate


# Authenticate with Google Translate service account key
client = translate.Client.from_service_account_json('../celestial-brand-444012-a9-2ab8d4732b4d.json')


path = './data/faq_khas_2.csv'
data  = pd.read_csv(path)


# Initialize RAG components
retriever = initialize_retriever("faiss_index")  # Path to FAISS index
generator = initialize_generator()


def extract_questions_and_answers(full_response):
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



print(data)

results = []

for index in range(2, 100):
    row = data.iloc[index]
    query = row["Question"]

    # Generate answer using RAG query handler

    # target_language = "en"
    # translated_content = client.translate(query, target_language=target_language)
    # translated_text = translated_content.get("translatedText")
    # logging.info(f"Translated input to English: {translated_text}")
    answer, question = rag_query_handler(query, retriever, generator)
    target_language = "mn"

    translated_answer_response = client.translate(answer, target_language=target_language)
    translated_answer = translated_answer_response.get("translatedText")

    translated_query_response = client.translate(query, target_language=target_language)
    translated_query = translated_query_response.get("translatedText")

    translated_real_response = client.translate(row['Answer'], target_language=target_language)
    translated_real = translated_real_response.get("translatedText")

    print(f"__________ Main Question: {translated_query}")
    print(f"__________ answer generated by RAG: {translated_answer}")

    # Store result
    results.append({
        "Question": translated_query,
        "answer": translated_answer,
        "real": row["Answer"], 
    })

# Save all results to a single CSV file
output_path = './data/model_answer_faq.csv'
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")


