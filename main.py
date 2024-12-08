from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import initialize_retriever, initialize_generator, rag_query_handler
from google.cloud import translate_v2 as translate
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize RAG components
retriever = initialize_retriever("faiss_index")  # Path to FAISS index
generator = initialize_generator()

# Authenticate with Google Translate service account key
client = translate.Client.from_service_account_json('./celestial-brand-444012-a9-2ab8d4732b4d.json')

@app.route('/')
def home():
    """
    Health check endpoint.
    """
    return 'RAG API is running!'

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for handling RAG queries.
    """
    # Parse incoming JSON request
    user_message = request.get_json()
    print(user_message)
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    content = user_message.get("content")
    if not content:
        return jsonify({"error": "Message content is required"}), 400
    try:
        target_language = "en"
        translated_content = client.translate(content, target_language=target_language)
        translated_text = translated_content.get("translatedText")
        logging.info(f"Translated input to English: {translated_text}")
        answer, question = rag_query_handler(translated_text, retriever, generator)
        target_language = "mn"
        translated_answer_response = client.translate(answer, target_language=target_language)
        translated_answer = translated_answer_response.get("translatedText")
        translated_question_response = client.translate(question, target_language=target_language)
        translated_question = translated_question_response.get("translatedText")
        return jsonify({"answer": translated_answer, "next_question": translated_question}), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=8888)
