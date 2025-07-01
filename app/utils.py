from fuzzywuzzy import process

def correct_spelling(input_text,symptoms_list_processed):
    best_match, score = process.extractOne(input_text, symptoms_list_processed.keys())
    return best_match if score >= 80 else None

def detect_intent(message):
    greeting_keywords = ["hello", "hi", "hey"]
    done_keywords = ["done", "that's it", "no more"]
    reset_keywords = ["reset", "start over"]
    
    if any(word in message for word in greeting_keywords):
        return "greeting"
    if any(word in message for word in done_keywords):
        return "done"
    if any(word in message for word in reset_keywords):
        return "reset"
    return "symptom"