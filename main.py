# -*- coding: utf-8 -*-
"""
Sci-Fi Code Improver Flask Application
Version: 1.1.4 (Refined error handling, regex, retry logic, removed unused config)

Improves code snippets using an LLM API, generates commit messages,
and provides a web interface.
"""

import os
import re
import json
import time
import datetime
import logging
import logging.handlers
import queue
from pathlib import Path
from urllib.parse import urljoin
from typing import Dict, List, Tuple, Optional, Union, Any

import bleach
import requests
from flask import Flask, render_template, request, jsonify, Response, session
from dotenv import load_dotenv

# --- Constants and Configuration ---
PROJECT_NAME = "sci-fi"
VERSION = "1.1.4"  # Incremented version
DEFAULT_COMMIT_MESSAGE = "chore: Code improved via LLM (API interaction issues)"
SLEEP_DURATION = 5
MAX_RETRIES = 3
IMPROVEMENTS_DIR_NAME = "improvements"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# --- Default Development Secret Key (INSECURE - CHANGE FOR PRODUCTION) ---
DEFAULT_DEV_SECRET_KEY = "dev_secret_key_please_change_in_prod"

load_dotenv()

# --- Configuration Loading with Type Hints and Error Handling ---
def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False, var_type: type = str) -> Any:
    """Gets an environment variable, optionally applies a type, and handles errors."""
    value = os.getenv(var_name, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{var_name}' not set.")
    if value is not None:
        try:
            return var_type(value)
        except ValueError as e:
            raise ValueError(f"Invalid type for environment variable '{var_name}': {e}") from e
    return value # Return None or the default if not required and not set

# --- Logging Setup ---
log_queue = queue.Queue()
queue_handler = logging.handlers.QueueHandler(log_queue)
# Configure root logger BEFORE loading config that might log warnings
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[logging.StreamHandler(), queue_handler])
logger = logging.getLogger(__name__)


# --- Load Configuration ---

# SECRET_KEY: Make it optional for easier local running, but WARN if default is used.
SECRET_KEY: str = get_env_var(
    "SECRET_KEY",
    default=DEFAULT_DEV_SECRET_KEY, # Provide the insecure default
    required=False,                 # Set to False
    var_type=str
)
if SECRET_KEY == DEFAULT_DEV_SECRET_KEY:
    logger.warning("=" * 60)
    logger.warning("WARNING: Using default insecure SECRET_KEY for development.")
    logger.warning("         Set the SECRET_KEY environment variable for secure sessions.")
    logger.warning("=" * 60)


# Optional configurations with defaults
MODEL_NAME: str = get_env_var("MODEL_NAME", "qwen2.5-coder-3b-instruct-mlx", var_type=str)
MAX_TOKENS: int = get_env_var("MAX_TOKENS", 2048, var_type=int)
TEMPERATURE: float = get_env_var("TEMPERATURE", 0.2, var_type=float)
CHUNK_SIZE_LINES: int = get_env_var("CHUNK_SIZE_LINES", 50, var_type=int)
# CONTEXT_WINDOW removed as it was unused
BASE_URL: str = get_env_var("BASE_URL", "http://localhost:1234", var_type=str).rstrip('/')
PROJECT_URL: str = get_env_var("PROJECT_URL", "https://github.com/fabriziosalmi/sci-fi", var_type=str)
LLM_API_TYPE: str = get_env_var("LLM_API_TYPE", "openrouter", var_type=str).lower()


# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = SECRET_KEY # Use the loaded (or default) secret key
IMPROVEMENTS_DIR: Path = Path(IMPROVEMENTS_DIR_NAME)
IMPROVEMENTS_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---

def load_prompts(filepath: str = "prompts.json") -> Dict[str, str]:
    """Loads prompts from a JSON file, handling errors gracefully."""
    default_prompts = {
        "improve_code": (
            "Improve the following {language} code snippet (chunk {chunk_number} of"
            " {total_chunks}). Focus on correctness, efficiency, readability, and modern practices.\n"
            "Previous context hint (last line of previous chunk, if any): {context_hint}\n" # Clarified hint meaning
            "Code:\n```{language}\n{code_chunk}\n```\n"
            "Return ONLY the improved code snippet for this chunk, preserving original functionality. "
            "Do NOT add explanations outside the code unless as comments."
        ),
        "commit_message": (
            "Generate a concise and descriptive Git commit message (following conventional commit format if possible, e.g., 'feat:', 'fix:', 'refactor:') "
            "for the following improved {language} code:\n\n{improved_code}\n\n"
            "Focus on the *changes* made by the improvement process."
        ),
    }
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            loaded_prompts = json.load(f)
            prompts_with_defaults = default_prompts.copy()
            prompts_with_defaults.update(loaded_prompts)
            return prompts_with_defaults
    except FileNotFoundError:
        logger.warning(f"Prompts file '{filepath}' not found. Using default prompts.")
        return default_prompts
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}. Using default prompts.")
        return default_prompts
    except Exception as e:
        logger.error(f"Unexpected error loading prompts from {filepath}: {e}. Using default prompts.")
        return default_prompts

prompts: Dict[str, str] = load_prompts()


class LLMAPIError(Exception):
    """Custom exception for LLM API errors."""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

def call_llm_api(
    prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    api_type: str = LLM_API_TYPE
) -> str:
    """
    Calls the LLM API using the requests library.
    Handles basic error checking and raises LLMAPIError on failure.
    """
    if api_type == "openai":
        endpoint = "v1/chat/completions"
    else: # Default to OpenAI compatible structure if not explicitly 'openai' (covers OpenRouter, LM Studio etc.)
        endpoint = "chat/completions" # Removed v1 prefix assumption for broader compatibility

    full_url = urljoin(base_url + '/', endpoint) # Ensure trailing slash on base_url for join

    headers = {"Content-Type": "application/json"}
    is_local_or_known_keyless = 'lmstudio' in base_url.lower() or ('localhost' in base_url.lower() and api_type == "openai")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif not is_local_or_known_keyless:
         logger.warning(f"API Key is empty/missing when calling {full_url}. This might be required for non-local/non-LM Studio services.")

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.debug(f"Calling LLM API: {full_url} with model {model_name}")
    response = None # Define response outside try for broader scope in error logging
    response_json = None # Define response_json outside try

    try:
        response = requests.post(full_url, headers=headers, json=data, timeout=120)
        response.raise_for_status() # Raises HTTPError for 4xx/5xx

        response_json = response.json()
        logger.debug(f"LLM API Raw Response: {response_json}")

        if "error" in response_json:
            error_detail = response_json['error']
            error_message = error_detail.get('message', 'Unknown API error structure') if isinstance(error_detail, dict) else str(error_detail)
            logger.error(f"LLM API returned an error: {error_message}")
            raise LLMAPIError(f"API Error: {error_message}", status_code=response.status_code)

        choices = response_json.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0 \
           or not isinstance(choices[0], dict) or not choices[0].get("message") \
           or not isinstance(choices[0]["message"], dict) or "content" not in choices[0]["message"]:
             logger.error(f"Unexpected LLM API response structure: {response_json}")
             raise LLMAPIError("Unexpected API response structure", status_code=response.status_code)

        content = choices[0]["message"]["content"]
        return str(content).strip() if content is not None else ""

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (4xx, 5xx) specifically raised by raise_for_status()
        logger.error(f"HTTP error {e.response.status_code} from LLM API: {e}")
        # Try to get more detail from response body if available
        error_detail_text = e.response.text
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_detail = error_json['error']
                error_detail_text = error_detail.get('message', str(error_detail)) if isinstance(error_detail, dict) else str(error_detail)
        except json.JSONDecodeError:
            pass # Use raw text if not JSON
        logger.error(f"LLM API Error Response Body: {error_detail_text[:500]}") # Log first 500 chars
        raise LLMAPIError(f"HTTP {e.response.status_code}: {error_detail_text}", status_code=e.response.status_code) from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Network request to LLM API failed: {e}")
        # Determine status code if possible (e.g., for connection errors, might be None)
        status_code = getattr(getattr(e, 'response', None), 'status_code', None)
        raise LLMAPIError(f"Network error contacting LLM API: {e}", status_code=status_code) from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from LLM API: {e}")
        response_text = response.text if response else "N/A"
        logger.error(f"Response text: {response_text[:500]}") # Log first 500 chars
        status_code = response.status_code if response else None
        raise LLMAPIError(f"Invalid JSON response from LLM API", status_code=status_code) from e
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Unexpected LLM API response structure or type: {e}. Response: {response_json}")
        status_code = response.status_code if response else None
        raise LLMAPIError(f"Unexpected API response structure or type: {e}", status_code=status_code) from e
    except LLMAPIError: # Re-raise if it's already the correct type
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during LLM API call")
        status_code = response.status_code if response else None
        raise LLMAPIError(f"An unexpected error occurred: {e}", status_code=status_code) from e


def call_llm_with_retry(
    prompt_generator: callable,
    api_key: str,
    *args,
    **kwargs
) -> str:
    """
    Calls the LLM API with retry logic using the provided prompt generator.
    Skips retries for certain client-side errors.
    Returns the result or raises LLMAPIError after max retries.
    """
    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            prompt = prompt_generator(*args, **kwargs)
            return call_llm_api(
                prompt, api_key, BASE_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE, LLM_API_TYPE
            )
        except LLMAPIError as e:
            logger.warning(f"LLM API call attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {e}")
            last_exception = e

            # --- Retry Strategy ---
            # Check for non-retryable client errors (4xx except 408 Timeout, 429 Rate Limit)
            # 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), 400 (Bad Request - often prompt/model issues)
            is_client_error = e.status_code and 400 <= e.status_code < 500
            is_retryable_client_error = e.status_code in [408, 429]

            if is_client_error and not is_retryable_client_error:
               logger.error(f"Non-retryable client error ({e.status_code}). Aborting retries.")
               raise last_exception # Abort immediately

            # If it's a server error (5xx) or a retryable client error, or network error, proceed to retry logic
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {SLEEP_DURATION} seconds...")
                time.sleep(SLEEP_DURATION)
            else:
                logger.error("Max retries reached for LLM API call.")
                raise last_exception # Raise the last exception after all retries fail

    # Should not be reached if MAX_RETRIES >= 0, but added for type hinting and safety
    raise last_exception if last_exception else LLMAPIError("LLM call failed after retries without specific exception.")


def improve_code_chunk(
    code_chunk: str,
    language: str,
    chunk_number: int,
    total_chunks: int,
    api_key: str,
    context_hint: str = "",
) -> str:
    """Improves a single chunk of code using the LLM API with retries."""
    logger.info(f"Improving chunk {chunk_number} of {total_chunks}...")

    def _generate_improve_prompt(lang, num, total, hint, chunk):
        return prompts["improve_code"].format(
            language=lang, chunk_number=num, total_chunks=total,
            context_hint=hint or "N/A", code_chunk=chunk,
        )

    try:
        improved_content = call_llm_with_retry(
            _generate_improve_prompt, api_key, language,
            chunk_number, total_chunks, context_hint, code_chunk
        )
        # Extract code *after* successful API call
        return extract_code_block(improved_content, language)
    except LLMAPIError as e:
        # Let process_code handle the overall failure, just return original chunk here
        logger.error(f"Failed to improve chunk {chunk_number} after retries: {e}. Returning original chunk.")
        return code_chunk


def generate_commit_message(improved_code: str, language: str, api_key: str) -> str:
    """Generates a commit message using the LLM API with retries."""
    logger.info("Generating commit message...")
    max_commit_context = 2000 # Limit context sent for commit message generation
    code_context = improved_code
    if len(improved_code) > max_commit_context:
        # Try to truncate at a newline
        trunc_point = improved_code.rfind('\n', 0, max_commit_context)
        if trunc_point == -1: # No newline found, just truncate hard
             trunc_point = max_commit_context
        code_context = improved_code[:trunc_point] + "\n... (code truncated for commit message context)"

    def _generate_commit_prompt(lang, code):
        return prompts["commit_message"].format(language=lang, improved_code=code)

    try:
        return call_llm_with_retry(_generate_commit_prompt, api_key, language, code_context)
    except LLMAPIError as e:
        # Let process_code handle the overall failure if needed, return default here
        logger.error(f"Failed to generate commit message after retries: {e}. Returning default message.")
        return DEFAULT_COMMIT_MESSAGE


def chunk_code_by_lines(code: str, max_lines: int = CHUNK_SIZE_LINES) -> List[str]:
    """Chunks code into blocks of roughly max_lines."""
    lines = code.splitlines()
    if not lines:
        return []
    chunks = []
    current_chunk_start = 0
    while current_chunk_start < len(lines):
        chunk_end = min(current_chunk_start + max_lines, len(lines))
        # Handle potential empty lines correctly by joining original lines
        chunk_lines = lines[current_chunk_start:chunk_end]
        chunks.append("\n".join(chunk_lines))
        current_chunk_start = chunk_end
    return chunks


def extract_code_block(text: str, language: Optional[str] = None) -> str:
    """
    Extracts code from the first markdown code block (```) found in the text.
    Handles optional language identifiers and empty blocks.
    If no block is found, returns the original text.
    """
    # Regex updated: Removed ^ and $ anchors to find the block anywhere.
    # Allows explanations before/after the block. Finds the *first* block.
    # Optional newline before closing ``` (\n?) remains.
    pattern = r"```(\w+)?\s*\n(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)

    if match:
        # block_language = match.group(1) # Language specified in the block (e.g., ```python) - useful for logging if needed
        code_content = match.group(2) or "" # The code inside the block, ensure string even if empty
        logger.debug(f"Extracted code block (language hint: {match.group(1) or 'None'})")
        return code_content.strip() # Strip leading/trailing whitespace from extracted code

    # If no markdown block is found, log a warning and return the original text, stripped.
    logger.warning("Could not find markdown code block (```...```) in LLM response. Returning raw response.")
    return text.strip()


def detect_language(code: str) -> str:
    """Rudimentary language detection based on common keywords and syntax."""
    if not isinstance(code, str) or not code.strip():
        return "python" # Default if input is not string or empty

    code_lower = code.lower()
    # More specific checks first
    if "package main" in code_lower and "func main" in code_lower: return "go"
    if "public static void main" in code and "class " in code_lower: return "java"
    if "import React" in code: return "javascript" # Could be jsx/tsx
    if "<template>" in code_lower and "<script" in code_lower: return "vue"
    if "#include" in code_lower and ("int main" in code or "void main" in code): return "cpp" # More likely C++ than C usually
    if "using System;" in code and "namespace " in code_lower: return "csharp"
    if "<?php" in code_lower: return "php"
    if "<!DOCTYPE html>" in code_lower or "<html>" in code_lower: return "html"
    if re.search(r'^\s*[\.#@]\w+\s*\{', code, re.MULTILINE): return "css" # Basic CSS/SCSS/LESS detection

    # Broader checks
    if "def " in code_lower and ":" in code and ("import " in code_lower or "from " in code_lower or "__name__" in code): return "python"
    if "function " in code_lower and ("const " in code_lower or "let " in code_lower or "var " in code_lower or "=>" in code): return "javascript"


    logger.debug("Could not reliably detect language, defaulting to 'python'")
    return "python"


def process_code(code: str, language: Optional[str], api_key: str) -> Tuple[str, str, Optional[Tuple[str, int]]]:
    """
    Processes the code: chunks, improves each chunk, combines, generates commit message.
    Returns (improved_code, commit_message, error_tuple | None).
    Error tuple is (error_message: str, suggested_status_code: int).
    """
    if not code or not code.strip():
        return "", "", ("No code provided.", 400) # Return error tuple for bad input

    original_language_request = language # Store original request
    detected_language = None
    if not language:
        language = detect_language(code)
        detected_language = language # Store detected language
        logger.info(f"No language provided, auto-detected: {language}")
    final_language = language # The language used for processing

    try:
        code_chunks = chunk_code_by_lines(code, CHUNK_SIZE_LINES)
        if not code_chunks:
             return "", "chore: No code processed.", None # No error, just nothing to do

        total_chunks = len(code_chunks)
        improved_code_parts: List[str] = []
        last_line_hint = ""
        improvement_failed_completely = False # Flag if *all* chunks fail or return original

        logger.info(f"Processing code in {total_chunks} chunk(s). Language: {final_language}")

        failed_chunk_count = 0
        for i, chunk in enumerate(code_chunks):
            if not chunk.strip(): # Skip processing for effectively empty chunks
                improved_code_parts.append(chunk)
                continue

            improved_chunk = improve_code_chunk(
                chunk, final_language, i + 1, total_chunks, api_key, context_hint=last_line_hint
            )
            # Check if improvement seems to have failed (returned original non-empty chunk)
            # Note: LLM might legitimately return the same code if it's already optimal.
            # This check is imperfect but useful for commit message generation.
            if improved_chunk == chunk:
                logger.warning(f"Improvement for chunk {i+1} seems to have failed or yielded no changes, using original.")
                failed_chunk_count += 1

            improved_code_parts.append(improved_chunk)

            # Update context hint based on the (potentially unimproved) chunk content
            lines = improved_chunk.strip().splitlines()
            last_line_hint = lines[-1].strip() if lines else ""

        improved_code = "\n".join(improved_code_parts)

        # Use default commit message if all non-empty chunks failed improvement
        improvement_failed_completely = (failed_chunk_count == len([c for c in code_chunks if c.strip()])) and failed_chunk_count > 0

        if improvement_failed_completely:
            commit_message = DEFAULT_COMMIT_MESSAGE
            logger.warning("Using default commit message as all chunks failed improvement or returned original content.")
        else:
            # Generate commit message based on the potentially partially improved code
            commit_message = generate_commit_message(improved_code, final_language, api_key)
            # If commit gen failed, it returns default, so no extra check needed here

        # Include detected language in response if it was detected
        result_info = {"language": final_language}
        if detected_language:
            result_info["detected_language"] = detected_language

        # Return code, commit message, and None for error
        # We might want to pass language info back differently later
        return improved_code, commit_message, None

    except LLMAPIError as e:
        error_message = f"LLM API Error during processing: {e}"
        logger.error(error_message)
        # Suggest 400 for client errors (config, prompt issues), 500 otherwise
        suggested_status = 400 if e.status_code and 400 <= e.status_code < 500 else 500
        return "", "", (error_message, suggested_status)
    except Exception as e:
        logger.exception("Unexpected error in process_code")
        error_message = f"Unexpected internal error during processing: {e}"
        return "", "", (error_message, 500) # Internal error -> 500


# --- Flask Routes ---

@app.route("/")
def index_route() -> str:
    """
    Renders the main index HTML page.

    Returns:
        str: Rendered HTML content for the index page.
    """
    return render_template("index.html",
                           project_name=PROJECT_NAME, version=VERSION,
                           model_name=MODEL_NAME, base_url=BASE_URL,
                           project_url=PROJECT_URL)

@app.route("/improve", methods=["POST"])
def improve() -> Union[Response, Tuple[Response, int]]: # Return type hint correction
    """
    Handles POST requests to improve a given code snippet.
    Expects JSON payload with 'code', optional 'language', and optional 'api_key'.
    Sanitizes input code, processes it through the LLM, saves results,
    and returns the improved code and commit message.

    Returns:
        Response: JSON response containing 'improved_code', 'commit_message', 'iteration', 'language'.
                  Status code 200 on success.
        Tuple[Response, int]: JSON response with 'error' message and status code 400 or 500 on failure.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request format. Expected JSON."}), 400

    code = data.get("code", "")
    language = data.get("language") # Can be None, process_code will detect
    api_key = data.get("api_key", "") # Default to empty string if not provided

    # Basic sanitization of code input (remove dangerous tags, keep basic structure)
    # Allow common code formatting tags if needed, but strip=True removes tags themselves,
    # keeping only content. If you need to preserve tags like <pre>, adjust bleach settings.
    # For pure code processing, stripping tags is safer.
    sanitized_code = bleach.clean(code, tags=[], attributes={}, strip=True)

    if not sanitized_code.strip(): # Check after sanitization
        # If original code had content but sanitized is empty, log it
        if code.strip():
            logger.warning("Provided code content was removed entirely by sanitization.")
        return jsonify({"error": "No valid code content provided after sanitization."}), 400

    # Check API key requirement (moved from call_llm_api for earlier check)
    is_local_or_known_keyless = ('lmstudio' in BASE_URL.lower() or ('localhost' in BASE_URL.lower() and LLM_API_TYPE == "openai"))
    if not api_key and not is_local_or_known_keyless:
         logger.warning("API key is missing or empty. This might be required for the configured LLM service.")
         # Consider returning 400 here? Or let the API call fail? Let API call fail for now.
         # return jsonify({"error": "API key is required for this service configuration."}), 400


    improved_code, commit_message, error_info = process_code(sanitized_code, language, api_key)

    if error_info:
        error_message, suggested_status_code = error_info
        # Use a more generic message for 500 errors to avoid leaking internal details
        public_error_msg = error_message
        if suggested_status_code == 500:
            public_error_msg = "An internal error occurred during code processing."
        elif suggested_status_code == 400:
             public_error_msg = f"Error processing request: {error_message}" # Provide more info for client errors

        return jsonify({"error": public_error_msg}), suggested_status_code

    # --- Success Path ---

    # Manage version history in session
    # NOTE: Storing full code in session can lead to large session sizes.
    # Consider storing references (e.g., filenames) if snippets are very large.
    history = session.get("version_history", [])
    if len(history) >= 10: # Limit history size
        history.pop(0)
    history.append(improved_code)
    session["version_history"] = history
    session.permanent = True # Make session last longer (default depends on Flask config)
    current_iteration = len(history)

    # Determine language used (either provided or detected)
    final_language = language or detect_language(sanitized_code) # Re-detect if needed for saving

    # Prepare and save improvement data to a file
    improvement_data = {
        "source_code": sanitized_code,
        "improved_code": improved_code,
        "commit_message": commit_message, # Save the raw commit message before attribution
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_used": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "chunk_size_lines": CHUNK_SIZE_LINES,
        "base_url": BASE_URL,
        "api_type": LLM_API_TYPE,
        "language": final_language,
        "iteration": current_iteration,
        "version": VERSION,
    }
    try:
        ts_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        filename = IMPROVEMENTS_DIR / f"improvement_{ts_str}_iter{current_iteration}.json"
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(improvement_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Improvement data saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save improvement data to file ({filename}): {e}", exc_info=True)
        # Don't fail the request, just log the error
    except Exception as e:
         logger.exception(f"Unexpected error saving improvement data to {filename}")
         # Don't fail the request


    # Add attribution to the commit message for display/use
    attribution = f"\n\n Improvement generated by [{PROJECT_NAME}]({PROJECT_URL}) v{VERSION} using model {MODEL_NAME}."
    full_commit_message = commit_message + attribution

    return jsonify({
        "improved_code": improved_code,
        "commit_message": full_commit_message,
        "iteration": current_iteration,
        "language": final_language # Return the language that was actually used
    }), 200

@app.route("/clear_session", methods=["POST"])
def clear_session() -> Tuple[Response, int]:
    """
    Clears the user's session data, specifically the 'version_history'.

    Returns:
        Tuple[Response, int]: JSON response indicating success and status code 200.
    """
    session.pop("version_history", None)
    logger.info("Session version_history cleared.")
    return jsonify({"status": "Session history cleared successfully."}), 200

@app.route("/detect_language", methods=["POST"])
def detect_language_route() -> Tuple[Response, int]:
    """
    Detects the programming language of a code snippet provided in the POST request body.
    Expects JSON payload with 'code'.

    Returns:
        Tuple[Response, int]: JSON response with 'language' (detected or default)
                              and 'detected' (boolean indicating if detection ran),
                              status code 200.
    """
    data = request.get_json()
    code = data.get("code", "") if data else ""
    detected = False
    final_lang = "python" # Default

    if code.strip(): # Only run detection if there's actual code content
        sanitized_code = bleach.clean(code, tags=[], attributes={}, strip=True)
        if sanitized_code.strip():
            final_lang = detect_language(sanitized_code)
            detected = True
            logger.info(f"Language detection request: Detected '{final_lang}'")
        else:
             logger.info("Language detection request: Code removed by sanitization, using default.")
    else:
        logger.info("Language detection request: No code provided, using default.")


    return jsonify({"language": final_lang, "detected": detected}), 200

@app.route("/stream")
def stream() -> Response:
    """
    Establishes a Server-Sent Events (SSE) stream to send log messages
    captured by the QueueHandler to the client. Includes heartbeats.

    Returns:
        Response: A Flask Response object configured for SSE (mimetype 'text/event-stream').
    """
    def generate():
        logger.info("SSE stream client connected.")
        # You could optionally send recent logs from the queue here if needed upon connection
        while True:
            try:
                # Block for a while waiting for a log record
                log_record = log_queue.get(timeout=15) # Increased timeout slightly
                log_entry = queue_handler.format(log_record)
                # Sanitize log entry slightly for SSE data field (newlines can break SSE)
                sanitized_log_entry = log_entry.replace('\n', '\\n')
                sse_message = f"data: {sanitized_log_entry}\n\n"
                yield sse_message
                log_queue.task_done() # Mark task as done for queue management
            except queue.Empty:
                # No logs within timeout, send a heartbeat comment to keep connection alive
                yield ": heartbeat\n\n"
            except GeneratorExit:
                # Client disconnected
                logger.info("SSE stream client disconnected.")
                return # Exit the generator
            except Exception as e:
                logger.error(f"Error in SSE generator: {e}", exc_info=True)
                # Yield an error message to the client? Maybe risky.
                # yield f"event: error\ndata: Error in log stream: {str(e)}\n\n"
                time.sleep(2) # Avoid tight loop on unexpected errors

    # Set headers for SSE
    response = Response(generate(), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no' # Important for Nginx proxying
    return response

# --- Main Execution ---
if __name__ == "__main__":
    # This check identifies the process started by the Werkzeug reloader
    is_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    # Log config only once, either by reloader's main process or if not debugging
    if is_reloader or not app.debug:
        logger.info(f"--- Starting {PROJECT_NAME} v{VERSION} ---")
        logger.info(f" * Flask Debug Mode: {app.debug}")
        logger.info("-" * 30)
        logger.info("Configuration:")
        logger.info(f"  SECRET_KEY Set: {'Yes (User Defined)' if SECRET_KEY != DEFAULT_DEV_SECRET_KEY else 'No (Using Default Dev Key - INSECURE)'}")
        logger.info(f"  API Endpoint Base: {BASE_URL}")
        logger.info(f"  API Type: {LLM_API_TYPE}")
        logger.info(f"  Model: {MODEL_NAME}")
        logger.info(f"  Temperature: {TEMPERATURE}")
        logger.info(f"  Max Tokens: {MAX_TOKENS}")
        logger.info(f"  Chunk Size (Lines): {CHUNK_SIZE_LINES}")
        logger.info(f"  Improvements Dir: {IMPROVEMENTS_DIR.resolve()}")
        logger.info(f"  Log Level: {logging.getLevelName(logger.getEffectiveLevel())}")
        logger.info("-" * 30)

    # Use host='0.0.0.0' to be accessible on the network
    # debug=False is set for this example, change to True for development features
    app.run(debug=False, host="0.0.0.0", port=7860)