from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
import json
import validators
import logging

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Llama LLM
llm = OllamaLLM(model="llama3")  # Specify the appropriate Llama model name

def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        cleaned_content = soup.get_text(separator="\n")
        cleaned_content = "\n".join(
            line.strip() for line in cleaned_content.splitlines() if line.strip()
        )
        return cleaned_content
    except requests.RequestException as e:
        logging.error(f"Error scraping webpage: {e}")
        return {"error": "Failed to retrieve webpage"}

def analyze_with_llama(content):
    prompt = f"""
    Based on the following website content, generate 3 questions with multiple-choice options to help categorize visitors based on their interests or industry.

    Content:
    {content}

    Format the response in JSON as follows:
    {{
        "questions": [
            {{
                "question": "Your question?",
                "options": ["Option 1", "Option 2", "Option 3"]
            }}
        ]
    }}
    """
    
    try:
        response_text = llm.invoke(prompt)  # Use invoke instead of predict
        logging.info(f"Llama API response: {response_text}")  # Log the raw response

        # Find the JSON part in the response
        start = response_text.find('{')
        if start != -1:
            json_part = response_text[start:]  # Extract the JSON part
            return json.loads(json_part)  # Parse the JSON part
        else:
            raise ValueError("No valid JSON found in the response")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}, Response: {response_text}")  # Log error if JSON parsing fails
        return {"error": "Failed to generate questions"}
    except Exception as e:
        logging.error(f"Llama API error: {e}")
        return {"error": "Failed to generate questions"}


def categorize_response(user_response):
    prompt = f"""
    Analyze this response and categorize the user: {user_response}
    """
    
    try:
        response_text = llm.invoke(prompt)  # Use invoke instead of predict
        return response_text.strip()  # Assuming Llama returns a string response
    except Exception as e:
        logging.error(f"Llama API error: {e}")
        return {"error": "Failed to categorize response"}

def is_valid_url(url):
    return validators.url(url)

@app.route('/generate-questions', methods=["POST"])
def generate_questions():
    url = request.json.get('url')
    if not is_valid_url(url):
        return jsonify({"error": "Invalid URL format"}), 400
    
    scraped_content = scrape_webpage(url)
    if "error" in scraped_content:
        return jsonify(scraped_content), 500
    questions_data = analyze_with_llama(scraped_content)
    if "error" in questions_data:
        return jsonify(questions_data), 500
    return jsonify({"questions": questions_data["questions"]})

@app.route('/categorize', methods=["POST"])
def categorize():
    user_response = request.json.get('user_response')
    category = categorize_response(user_response)
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(debug=True)