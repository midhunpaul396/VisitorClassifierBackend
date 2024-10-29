from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import openai
import os
from dotenv import load_dotenv
import json


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")

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
        print(f"Error scraping webpage: {e}")
        return {"error": "Failed to retrieve webpage"}

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


def categorize_response(user_response):
    messages = [{"role": "user", "content": f"Analyze this response and categorize the user: {user_response}"}]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {"error": "Failed to categorize response"}


@app.route('/generate-questions', methods=["POST"])
def generate_questions():
    url = request.json.get('url')
    scraped_content = scrape_webpage(url)
    if "error" in scraped_content:
        return jsonify(scraped_content), 500
    questions_data = analyze_with_chatgpt(scraped_content)
    if "error" in questions_data:
        return jsonify(questions_data), 500
    # Return the inner questions array directly
    return jsonify({"questions": questions_data["questions"]})

@app.route('/categorize', methods=["POST"])
def categorize():
    user_response = request.json.get('user_response')
    category = categorize_response(user_response)
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(debug=True)
