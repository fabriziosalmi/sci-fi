from flask import Flask, render_template, request, jsonify, Response, session
import os
import re
import json
import time
import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import bleach
import logging
from pathlib import Path
import subprocess  # Import the subprocess module
from urllib.parse import urljoin  # added import

# --- Constants and Configuration ---
PROJECT_NAME = "sci-fi"
VERSION = "1.0.3"
DEFAULT_COMMIT_MESSAGE = "chore: No changes (API error)"
SLEEP_DURATION = 5

load_dotenv()

# --- Constants with Type Hints and Error Handling ---
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5-coder-3b-instruct-mlx")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.2))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", 10))
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:1234")
PROJECT_URL: str = os.getenv("PROJECT_URL", "https://github.com/fabriziosalmi/sci-fi")
MAX_RETRIES: int = 3
IMPROVEMENTS_DIR: str = "improvements"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "mysecret")
if not app.secret_key:
    raise ValueError("SECRET_KEY environment variable not set.")

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Classes ---
class DebugLogger:
    """A simple logger that outputs messages in a format suitable for SSE."""

    def __init__(self, level: str = "INFO"):
        self.logs: List[str] = []
        self.level = level  # DEBUG, INFO, WARNING, ERROR
        self.levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    def _log(self, message: str, level: str) -> str:
        """Logs a message with a timestamp and level, if the level is appropriate."""
        if self._should_log(level):
            timestamp = datetime.datetime.now().isoformat()
            formatted_message = f"{timestamp} - {level}: {message}"
            self.logs.append(formatted_message)
            return f"data: {formatted_message}\n\n"  # SSE format

    def write(self, message: str) -> str:
        """Logs a message with INFO level."""
        return self._log(message, "INFO")

    def clear(self) -> str:
        """Clears the log buffer."""
        self.logs = []
        return "data: \n\n"

    def debug(self, message: str) -> str:
        """Logs a message with DEBUG level."""
        return self._log(message, "DEBUG")

    def info(self, message: str) -> str:
        """Logs a message with INFO level."""
        return self._log(message, "INFO")

    def warning(self, message: str) -> str:
        """Logs a message with WARNING level."""
        return self._log(message, "WARNING")

    def error(self, message: str) -> str:
        """Logs a message with ERROR level."""
        return self._log(message, "ERROR")

    def _should_log(self, level: str) -> bool:
        """Checks if the given level should be logged based on the logger's configured level."""
        return self.levels.index(level) >= self.levels.index(self.level)

logger = DebugLogger(level="DEBUG")

# --- Helper Functions ---
def load_prompts(filepath: str = "prompts.json") -> Dict[str, str]:
    """Loads prompts from a JSON file, handling errors gracefully."""
    default_prompts = {
        "improve_code": (
            "Improve the following {language} code snippet (chunk {chunk_number} of"
            " {total_chunks}).\n{truncation_info}\nCode:\n```\n{code_chunk}\n```"
        ),
        "commit_message": (
            "Generate a concise commit message for the following improved {language}"
            " code:\n{improved_code}"
        ),
    }
    try:
        with open(filepath, "r") as f:
            prompts = json.load(f)
            return {
                key: prompts.get(key, default_prompts[key]) for key in default_prompts
            }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(logger.error(f"Error loading prompts from {filepath}: {e}. Using defaults."))
        return default_prompts

prompts: Dict[str, str] = load_prompts()


def run_curl_command(prompt: str, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float) -> str:
    """
    Executes a curl command to interact with the OpenAI-compatible API.
    Returns the extracted content from the response, or an error message.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Combine BASE_URL with endpoint using urljoin
    full_url = urljoin(base_url.rstrip('/') + '/', "chat/completions")

    command = [
        "curl",
        "-X", "POST",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data),
        full_url,
    ]

    print(logger.info(f"Executing curl command: {' '.join(command)}"))

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(logger.debug(f"Curl output: {result.stdout}")) # Log the full output
        response_json = json.loads(result.stdout)

        # Check for API errors in the response
        if "error" in response_json:
            error_message = response_json['error'].get('message', 'Unknown API error')
            print (logger.error(f"API Error: {error_message}"))
            return f"API Error: {error_message}"


        return response_json["choices"][0]["message"]["content"].strip()

    except subprocess.CalledProcessError as e:
        print(logger.error(f"Curl command failed: {e}"))
        print(logger.error(f"Stderr: {e.stderr}"))  # Print stderr for more details
        return f"Curl command failed: {e.stderr}"
    except json.JSONDecodeError as e:
        print(logger.error(f"Failed to decode JSON response: {e}"))
        print(logger.error(f"Response text: {result.stdout}")) #raw response
        return f"Failed to decode JSON response: {e}"
    except (KeyError, IndexError) as e:
        print (logger.error(f"Unexpected response structure: {e}"))
        return f"Unexpected response structure: {e}"
    except Exception as e:
        print(logger.error(f"An unexpected error occurred: {e}"))
        return f"An unexpected error occurred: {e}"


def improve_code_chunk(
    code_chunk: str,
    language: str,
    chunk_number: int,
    total_chunks: int,
    truncation_info: str = "",
    retry: int = 0,
) -> str:
    """Improves a single chunk of code using curl, with retries."""
    prompt = prompts["improve_code"].format(
        language=language,
        chunk_number=chunk_number,
        total_chunks=total_chunks,
        truncation_info=truncation_info,
        code_chunk=code_chunk,
    )
    print(logger.info(f"Improving chunk {chunk_number} of {total_chunks} (attempt {retry + 1})..."))

    try:
        response_content = run_curl_command(prompt, OPENROUTER_API_KEY, BASE_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE)
        if "API Error" in response_content or "Curl command failed" in response_content or "Failed to decode JSON" in response_content:
            if retry < MAX_RETRIES:
                time.sleep(SLEEP_DURATION)
                print (logger.info("Retrying..."))
                return improve_code_chunk(code_chunk, language, chunk_number, total_chunks, truncation_info, retry + 1)
            else:
                print (logger.warning("Max retries reached. Returning original code chunk."))
                return code_chunk #return original
        return response_content

    except Exception as e:
        print(logger.error(f"Unexpected error in improve_code_chunk: {e}"))
        if retry < MAX_RETRIES:
           time.sleep(SLEEP_DURATION)
           return improve_code_chunk(code_chunk, language, chunk_number, total_chunks, truncation_info, retry+1)

        return code_chunk


def generate_commit_message(improved_code: str, language: str, retry: int = 0) -> str:
    """Generates a commit message using curl, with retries."""
    prompt = prompts["commit_message"].format(
        language=language, improved_code=improved_code
    )
    print(logger.info(f"Generating commit message (attempt {retry + 1})..."))

    try:
        response_content = run_curl_command(prompt, OPENROUTER_API_KEY, BASE_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE)
        if "API Error" in response_content or "Curl command failed" in response_content or "Failed to decode JSON" in response_content:
          if retry < MAX_RETRIES:
              time.sleep(SLEEP_DURATION)
              print(logger.info("retrying..."))
              return generate_commit_message(improved_code, language, retry + 1)
          else:
              print(logger.warning("Max retries reached. Returning default commit message"))
              return DEFAULT_COMMIT_MESSAGE
        return response_content

    except Exception as e:
        print(logger.error(f"Error generating commit message: {e}"))
        if retry < MAX_RETRIES:
           time.sleep(SLEEP_DURATION)
           return generate_commit_message(improved_code, language, retry + 1)

        return DEFAULT_COMMIT_MESSAGE



def chunk_code(code: str, max_lines: int = CHUNK_SIZE) -> List[Tuple[str, str]]:
    """Splits the code into chunks, handling truncation intelligently."""
    lines = code.splitlines()
    chunks: List[Tuple[str, str]] = []
    current_chunk: List[str] = []

    for i, line in enumerate(lines):
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            trunc_point = _find_truncation_point(current_chunk)
            chunks.append(("\n".join(current_chunk[:trunc_point]),
                           _generate_truncation_message(current_chunk, trunc_point)))
            current_chunk = current_chunk[trunc_point:]  # Keep context

    if current_chunk:
        chunks.append(("\n".join(current_chunk), ""))

    return chunks

def _find_truncation_point(chunk: List[str]) -> int:
    """Finds a suitable point for code truncation."""
    for i in range(len(chunk) - 1, max(len(chunk) - CONTEXT_WINDOW - 1, -1), -1):
        if re.match(r"^(def|class)\s+(\w+)", chunk[i]):
            return i
    return len(chunk)

def _generate_truncation_message(chunk: List[str], trunc_point: int) -> str:
    """Generates an informative truncation message."""
    if trunc_point < len(chunk):
        match = re.match(r"^(def|class)\s+(\w+)", chunk[trunc_point])
        if match:
            return f"# TRUNCATED AT: {match.group(1)} {match.group(2)}()"
    return ""

def process_code(code: str, language: str) -> Tuple[str, str, str]:
    """Processes the code, improves it, and generates a commit message."""
    if not code or not language:
        return "", "", "Please enter code and select a language."

    try:
        # No more client = get_openai_client()
        chunks = chunk_code(code)
        total_chunks = len(chunks)
        improved_code = ""
        truncation_info = ""

        for i, (chunk, marker) in enumerate(chunks):
            try:
                improved_chunk = improve_code_chunk(
                     chunk, language, i + 1, total_chunks, truncation_info
                )
                if i < total_chunks - 1:
                    if re.search(r"^(def|class)\s+", improved_chunk, re.MULTILINE):
                        print(logger.warning(f"Possible structure issue in chunk {i+1}. Skipping."))
                        continue

                improved_code += improved_chunk + "\n"
                truncation_info = f"Previous chunk ended with: {marker}" if marker else ""

            except Exception as e:
                return "", "", str(e)  # Catch any errors during chunk improvement

        try:
            commit_message = generate_commit_message(improved_code, language)
        except Exception as e:
             return improved_code, DEFAULT_COMMIT_MESSAGE, str(e)

        return improved_code, commit_message, ""

    except Exception as e:
        error_message = f"Unexpected error in process_code: {e}"
        print(logger.error(error_message))
        return "", "", error_message  # Return a specific error message



def extract_code_block(text: str, language: Optional[str] = None) -> str:
    """Extracts code blocks from the given text."""
    if language:
        pattern_specific = rf"```{language}\s*\n(.*?)\n```"
        match = re.search(pattern_specific, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern_generic = r"```(?!(?:\w+\n))\s*\n(.*?)\n```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return text.strip()

    pattern_any = r"```(?:\w+)?\s*\n(.*?)\n```"
    match_any = re.search(pattern_any, text, re.DOTALL)
    if match_any:
        return match_any.group(1).strip()

    return text.strip()


def detect_language(code: str) -> str:
    """Detects the programming language."""
    code_lower = code.lower()
    if any(keyword in code_lower for keyword in ["import ", "from "]) and "def " in code_lower:
        return "python"
    elif "function " in code_lower and "{" in code_lower:
        return "javascript"
    elif "public " in code_lower and "class " in code_lower:
        return "java"
    elif "package main" in code_lower and "func " in code_lower:
        return "go"
    return "python"  # Default


# --- Flask Routes ---
@app.route("/")
def index() -> str:
    """Renders the main index page."""
    return render_template("index.html")

@app.route("/improve", methods=["POST"])
def improve() -> Tuple[Response, int] | Tuple[jsonify, int]:
    """Handles the code improvement request, returning specific error codes."""
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "No code provided"}), 400

    code = data.get("code")
    language = data.get("language", "python")

    allowed_tags = ["b", "i", "u", "em", "strong", "a", "pre", "code", "br"]
    allowed_attributes = {"a": ["href", "title"]}
    code = bleach.clean(code, tags=allowed_tags, attributes=allowed_attributes)

    improved_code, commit_message, error = process_code(code, language)
    if error:
        #  Return 400 for API-related errors, 500 for other internal errors
        if "Unexpected error in process_code" in error:
          return jsonify({"error": error}), 500
        else:
          return jsonify({"error": error}), 400


    history = session.get("version_history", [])
    if len(history) >= 10:
        history.pop(0)
    history.append(improved_code)
    session["version_history"] = history

    improvement_data = {
        "source_code": code,
        "improved_code": improved_code,
        "commit_message": commit_message,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_used": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "chunk_size": CHUNK_SIZE,
        "context_window": CONTEXT_WINDOW,
        "base_url": BASE_URL,
        "iteration": len(history),
    }
    improvements_dir_path = Path(IMPROVEMENTS_DIR)
    improvements_dir_path.mkdir(exist_ok=True)
    filename = improvements_dir_path / f"improvement-{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(improvement_data, f)

    commit_message = (
        commit_message
        + f"\n\nImprovement generated by **[{PROJECT_NAME}]({PROJECT_URL})** using {MODEL_NAME}"
    )
    return (
        jsonify(
            {
                "improved_code": extract_code_block(improved_code, language),
                "commit_message": commit_message,
                "iteration": len(history),
            }
        ),
        200,
    )

@app.route("/clear_session", methods=["POST"])
def clear_session() -> Tuple[jsonify, int]:
    """Clears the session data."""
    session.clear()
    return jsonify({"status": "Session cleared."}), 200

@app.route("/detect_language", methods=["POST"])
def detect_language_route() -> Tuple[jsonify, int]:
    """Detects the language of the provided code snippet."""
    data = request.get_json()
    code = data.get("code", "") if data else ""
    if not code:
        return jsonify({"language": "python"}), 200

    code = bleach.clean(code)
    return jsonify({"language": detect_language(code)}), 200

@app.route("/stream")
def stream() -> Response:
    """Provides a stream for server-sent events (SSE)."""
    def generate():
        while True:
            if logger.logs:
                yield logger.logs.pop(0)
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of current directory: {os.listdir()}")
        # New prints for API endpoint, model and options
        print("API Endpoint:", BASE_URL)
        print("Model:", MODEL_NAME)
        print("Temperature:", TEMPERATURE)
        print("Max Tokens:", MAX_TOKENS)
        print("Chunk Size:", CHUNK_SIZE)
        print("Context Window:", CONTEXT_WINDOW)
    app.run(debug=True, host="0.0.0.0", port=7860)