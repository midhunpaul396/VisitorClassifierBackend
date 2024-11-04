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
import openai

load_dotenv()

app = Flask(__name__)

# Set up CORS with specific options
CORS(app, resources={r"/*": {
    "origins": "http://localhost:3000",
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

openai.api_key = os.getenv("OPENAI_API_KEY")

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
    

def analyze_with_chatgpt(content):
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        response_text = response.choices[0].message['content'].strip()
        return json.loads(response_text)  # Parse directly to JSON
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {"error": "Failed to generate questions"}

# Function for categorizing responses with Ollama
def categorize_response_with_llama(user_response):
    prompt = f"Analyze this response and categorize the user: {user_response}"
    try:
        response_text = llm.invoke(prompt)
        return response_text.strip()
    except Exception as e:
        logging.error(f"Llama API error: {e}")
        return {"error": "Failed to categorize response"}

# Function for categorizing responses with OpenAI
def categorize_response_with_chatgpt(user_response):
    messages = [{"role": "user", "content": f"Analyze this response and categorize the user: {user_response}"}]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return {"error": "Failed to categorize response"}

def is_valid_url(url):
    return validators.url(url)

@app.route('/generate-questions', methods=["POST"])
def generate_questions():
    logging.info(f"Received request: {request.json}")  # Log the incoming request
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

@app.route('/generate-openai-questions', methods=["POST"])
def generate_open_ai_questions():
    url = request.json.get('url')
    scraped_content = scrape_webpage(url)
    if "error" in scraped_content:
        return jsonify(scraped_content), 500
    questions_data = analyze_with_chatgpt(scraped_content)
    if "error" in questions_data:
        return jsonify(questions_data), 500
    # Return the inner questions array directly
    return jsonify({"questions": questions_data["questions"]})

# Route for categorizing user response with Ollama
@app.route('/categorize-ollama', methods=["POST"])
def categorize_ollama():
    user_response = request.json.get('user_response')
    category = categorize_response_with_llama(user_response)
    return jsonify({"category": category})

# Route for categorizing user response with OpenAI
@app.route('/categorize-openai', methods=["POST"])
def categorize_openai():
    user_response = request.json.get('user_response')
    category = categorize_response_with_chatgpt(user_response)
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(debug=True)
