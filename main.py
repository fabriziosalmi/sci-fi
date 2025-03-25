from flask import Flask, render_template, request, jsonify, Response, session  # Added session import
import openai
import os
import re
import json
import time
import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import bleach
import pytest  # Import pytest
import logging

# Project name and version
PROJECT_NAME = "sci-fi"  # Simple Code Improvement Framework & Interface
VERSION = "1.0.0"

# Initialize standard logging
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s: %(message)s",
)

# Module docstring
"""
Main module for Code Improver web application.

This module configures the Flask app, loads prompts, handles code improvement
via OpenAI interactions, and provides multiple routes for API and testing.
"""

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemini-2.0-pro-exp-02-05:free")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 10))
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
PROJECT_URL = os.getenv("PROJECT_URL", "https://github.com/fabriziosalmi/sci-fi")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "mysecret")  # Added: enable sessions

# --- Logger (Server-Sent Events with Timestamps and Levels) ---
class DebugLogger:
    def __init__(self, level="INFO"):
        self.logs: List[str] = []
        self.level = level  # Log level (DEBUG, INFO, WARNING, ERROR)

    def _log(self, message: str, level: str) -> str:
        if self._should_log(level):  # Check if we should log based on the level
            timestamp = datetime.datetime.now().isoformat()
            formatted_message = f"{timestamp} - {level}: {message}"
            self.logs.append(formatted_message)
            return f"data: {formatted_message}\n\n"  # SSE format
        return ""

    def write(self, message: str) -> str:
        #  Default to INFO level if no level is specified.
        return self._log(message, "INFO")

    def clear(self) -> str:
        self.logs = []
        return "data: \n\n"

    def debug(self, message: str) -> str:
        return self._log(message, "DEBUG")

    def info(self, message: str) -> str:
        return self._log(message, "INFO")

    def warning(self, message: str) -> str:
        return self._log(message, "WARNING")

    def error(self, message: str) -> str:
        return self._log(message, "ERROR")

    def _should_log(self, level: str):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        return levels.index(level) >= levels.index(self.level)

logger = DebugLogger(level="DEBUG")  # Set the desired log level here

# --- Helper Functions ---

def load_prompts(filepath: str = "prompts.json") -> Dict[str, str]:
    default_prompts = {
        "improve_code": """Improve the following {language} code snippet (chunk {chunk_number} of {total_chunks}).
{truncation_info}
Code:
```
{code_chunk}
```""",
        "commit_message": """Generate a concise commit message for the following improved {language} code:
{improved_code}""",
    }
    try:
        with open(filepath, "r") as f:
            prompts = json.load(f)
            # Use .get() with defaults to handle missing keys
            return {key: prompts.get(key, default_prompts[key]) for key in default_prompts}

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(logger.error(f"Error loading prompts from {filepath}: {e}. Using defaults."))
        return default_prompts


prompts = load_prompts()


def get_openai_client():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    print(logger.info("Initializing OpenAI client..."))
    client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)
    print(logger.info("OpenAI client initialized."))
    return client

MAX_RETRIES = 3  # New retry limit

def improve_code_chunk(client, code_chunk: str, language: str, chunk_number: int, total_chunks: int, truncation_info: str = "", retry: int = 0) -> str:
    """
    Improve a chunk of code via OpenAI API.

    Params:
        client: The OpenAI client instance.
        code_chunk: The code snippet to improve.
        language: Programming language of the code.
        chunk_number: Index of the current chunk (1-indexed).
        total_chunks: Total number of chunks.
        truncation_info: Info regarding any previously truncated sections.

    Returns:
        The improved code as a string.
    """
    prompt = prompts["improve_code"].format(
        language=language, chunk_number=chunk_number, total_chunks=total_chunks,
        truncation_info=truncation_info, code_chunk=code_chunk
    )
    print(logger.info(f"Improving chunk {chunk_number} of {total_chunks}..."))
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        #  Check for empty response
        if not response.choices:
            raise ValueError("Empty response from OpenAI API.")
        return response.choices[0].message.content

    except openai.RateLimitError as e:
        logging.error(f"Rate limit exceeded: {e}")
        if retry < MAX_RETRIES:
            time.sleep(5)
            return improve_code_chunk(client, code_chunk, language, chunk_number, total_chunks, truncation_info, retry + 1)
        else:
            raise

    except openai.APIError as e:
        print(logger.error(f"OpenAI API Error: {e}"))
        raise

    except Exception as e:
        print(logger.error(f"Unexpected error in improve_code_chunk: {e}"))
        raise



def generate_commit_message(client, improved_code: str, language: str, retry: int = 0) -> str:
    prompt = prompts["commit_message"].format(language=language, improved_code=improved_code)
    print(logger.info("Generating commit message..."))
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        if not response.choices:
             raise ValueError("Empty response from OpenAI API.")
        return response.choices[0].message.content

    except openai.RateLimitError as e:
        print(logger.error(f"Rate limit exceeded: {e}"))
        if retry < MAX_RETRIES:
            time.sleep(5)
            return generate_commit_message(client, improved_code, language, retry + 1)
        else:
            raise

    except openai.APIError as e:
        print(logger.error(f"OpenAI API Error: {e}"))
        raise

    except Exception as e:
        print(logger.error(f"Error generating commit message: {e}"))
        raise


def chunk_code(code: str, max_lines: int = CHUNK_SIZE) -> List[Tuple[str, str]]:
    lines = code.splitlines()
    chunks: List[Tuple[str, str]] = []
    current_chunk: List[str] = []

    for i, line in enumerate(lines):
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            trunc_point = ""
            # Iterate backwards from the end of the chunk
            for j in range(len(current_chunk) - 1, max(len(current_chunk) - CONTEXT_WINDOW - 1, -1), -1):
                #  Check for function or class definition
                if match := re.match(r"^(def|class)\s+(\w+)", current_chunk[j]):
                    trunc_point = f"# TRUNCATED AT: {match.group(1)} {match.group(2)}()"  #  More informative message
                    current_chunk = current_chunk[:j] # Truncate before the definition
                    break
            chunks.append(("\n".join(current_chunk), trunc_point))
            current_chunk = []

    if current_chunk:  #  Handle the last chunk
        chunks.append(("\n".join(current_chunk), ""))

    return chunks


def process_code(code: str, language: str) -> Tuple[str, str, str]:
    if not code or not language:
        return "", "", "Please enter code and select a language."

    try:
        client = get_openai_client()
        chunks = chunk_code(code)
        total_chunks = len(chunks)
        improved_code = ""
        truncation_info = ""

        for i, (chunk, marker) in enumerate(chunks):
            try:
                improved_chunk = improve_code_chunk(client, chunk, language, i + 1, total_chunks, truncation_info)
                # Improved Structure Check and handling
                if i < total_chunks - 1:  # Only check if it's *not* the last chunk
                    if re.search(r"^(def|class)\s+", improved_chunk, re.MULTILINE):
                       print(logger.warning(f"Warning: Possible incorrect structure detected in chunk {i + 1}.  Skipping chunk."))
                       continue  # Skip this chunk

                improved_code += improved_chunk + "\n"
                truncation_info = f"Previous response ended with: {marker}" if marker else ""
            except Exception as e:
                return "", "", str(e)

        commit_message = generate_commit_message(client, improved_code, language)
        return improved_code, commit_message, ""

    except Exception as e:
        print(logger.error(f"Unexpected error in process_code: {e}"))
        return "", "", str(e)


def extract_code_block(text: str, language: str = "python") -> str:
    # Try to match fenced code with provided language
    match = re.search(rf"```{language}\s*\n(.*?)\s*```", text, re.DOTALL)
    if match:
         return match.group(1).strip()
    # Fallback: remove any generic fenced code block
    generic_match = re.search(r"```(?:\w+)?\s*\n(.*?)\s*```", text, re.DOTALL)
    if generic_match:
         return generic_match.group(1).strip()
    return text.strip()


def detect_language(code: str) -> str:
    # More robust language detection
    code_lower = code.lower()  # Case-insensitive matching
    if any(keyword in code_lower for keyword in ["import ", "from "]) and "def " in code_lower:
        return "python"
    elif "function " in code_lower and "{" in code_lower:
        return "javascript"
    elif "public " in code_lower and "class " in code_lower:
         return "java"
    elif "package main" in code_lower and "func " in code_lower:
        return "go"
    else:
        return "python"  # Default to Python


# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/improve", methods=["POST"])
def improve():
    data = request.get_json()
    code = data.get("code")
    language = data.get("language", "python")  # Default to Python

    if not code:
        return jsonify({"error": "No code provided"}), 400

    # Sanitize the code input using bleach, allowing specific tags/attributes
    allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'a', 'pre', 'code', 'br']
    allowed_attributes = {'a': ['href', 'title']}

    code = bleach.clean(code, tags=allowed_tags, attributes=allowed_attributes)

    improved_code, commit_message, error = process_code(code, language)
    if error:
        return jsonify({"error": error}), 500

    # Versioning: track up to 10 iterations in session
    history = session.get("version_history", [])
    if len(history) >= 10:
        history.pop(0)  # Remove oldest entry to allow new iteration instead of erroring
    history.append(improved_code)
    session["version_history"] = history

    # New dataset logging step with additional process tracking info:
    from pathlib import Path
    improvement_data = {
        "source_code": code,
        "improved_code": improved_code,
        "commit_message": commit_message.strip(),
        "timestamp": datetime.datetime.now().isoformat(),
        "model_used": MODEL_NAME,           # Added model info
        "temperature": TEMPERATURE,         # Added temperature info
        "max_tokens": MAX_TOKENS,           # Added max tokens
        "chunk_size": CHUNK_SIZE,           # Added chunk size
        "context_window": CONTEXT_WINDOW,   # Added context window
        "base_url": BASE_URL,               # Added base_url info
        "iteration": len(history)           # Added iteration number info
    }
    improvements_dir = Path("improvements")
    improvements_dir.mkdir(exist_ok=True)
    filename = improvements_dir / f"improvement-{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(improvement_data, f)
    
    # Append commit message with model usage info.
    commit_message = commit_message.strip() + "\n\nImprovement generated by [sci-fi](https://github.com/fabriziosalmi/sci-fi) using " + MODEL_NAME
    return jsonify({
        "improved_code": extract_code_block(improved_code, language),
        "commit_message": commit_message,
        "iteration": len(history)
    })

# New route to clear session version history
@app.route("/clear_session", methods=["POST"])
def clear_session():
    session.clear()
    return jsonify({"status": "Session cleared."})

@app.route("/detect_language", methods=["POST"])
def detect_language_route():
    code = request.get_json().get("code", "")
    if not code: # Check for empty code
        return jsonify({"language": "python"}), 200 # Return default, but no error

     # Sanitize the code input using bleach
    code = bleach.clean(code)
    return jsonify({"language": detect_language(code)})


@app.route("/stream")
def stream():
    def generate():
        while True:
            if logger.logs:
                yield logger.logs.pop(0)
            time.sleep(0.5)  # Adjust sleep time as needed

    return Response(generate(), mimetype='text/event-stream')

# --- Basic Tests (using pytest) ---
def test_chunk_code():
    test_code = "line1\nline2\nline3\nline4\nline5\ndef my_function():\n    pass\nline8"
    chunks = chunk_code(test_code, max_lines=3)
    assert len(chunks) == 3
    assert chunks[0][0] == "line1\nline2\nline3"
    assert chunks[1][0] == "line4\nline5"
    assert "TRUNCATED AT: def my_function()" in chunks[1][1]
    assert chunks[2][0] == "def my_function():\n    pass\nline8"
    assert chunks[2][1] == ""

    # Test empty code
    assert chunk_code("") == [('', '')]

    # Test code shorter than max_lines
    test_code_short = "short_line1\nshort_line2"
    chunks_short = chunk_code(test_code_short, max_lines=5)
    assert len(chunks_short) == 1
    assert chunks_short[0][0] == test_code_short
    assert chunks_short[0][1] == ""

def test_detect_language():
    assert detect_language("import os\nfrom sys import argv") == "python"
    assert detect_language("function myFunction() {}") == "javascript"
    assert detect_language("public class MyClass {}") == "java"
    assert detect_language("package main\nfunc main() {}") == "go"
    assert detect_language("some random text") == "python"  # Default case
    assert detect_language("") == "python"
    assert detect_language("Import OS\nFrom SYS import Argv") == "python"

def test_extract_code_block():
     assert extract_code_block("```python\nprint('Hello')\n```") == "print('Hello')"
     assert extract_code_block("Some text\n```javascript\nconsole.log('Hi');\n```\nMore text") == "console.log('Hi');"
     assert extract_code_block("No code block here") == "No code block here"
     assert extract_code_block("```python\n```") == ""  # Empty code block
     assert extract_code_block("```\nprint('Hello')\n```", "python") == "```\nprint('Hello')\n```"


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir()}")
    app.run(debug=True, host="0.0.0.0", port=7860)
