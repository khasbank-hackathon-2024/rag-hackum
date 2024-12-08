
target_language = "mn"
translated_answer_response = client.translate(answer, target_language=target_language)
translated_answer = translated_answer_response.get("translatedText")
translated_question_response = client.translate(question, target_language=target_language)
translated_question = translated_question_response.get("translatedText")