def final_prediction(qna_result, image_result, model, uploaded_file):
    combined_result = qna_result  # Modify this based on your actual logic
    
    prediction = model.predict([combined_result])
    confidence = max(model.predict_proba([combined_result])[0])
    return prediction[0], confidence
