# model.py
def predict_policy(text: str):
    # Dummy model – replace with your actual ML model logic
    if "education" in text.lower():
        return "Education Policy"
    elif "health" in text.lower():
        return "Healthcare Policy"
    elif "farm" in text.lower() or "crop" in text.lower():
        return "Agriculture Policy"
    elif "women" in text.lower():
        return "Women Empowerment Policy"
    elif "job" in text.lower() or "employment" in text.lower():
        return "Employment Policy"
    else:
        return "General Policy"
