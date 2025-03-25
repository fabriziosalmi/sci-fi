import unittest
from unittest.mock import patch, MagicMock, call
import json
import os
import requests  # <-- IMPORTED requests
from pathlib import Path
import shutil
import sys
import time

# --- Set required environment variables BEFORE importing the main module ---
os.environ['SECRET_KEY'] = 'testing_secret_key_123'

# Ensure the main module directory is in the path if running tests from a different location
# Or structure your project with tests outside the main package.
# For simplicity here, assuming tests are run from the same directory as main.py
import main  # Import the updated main module

# Import specific items needed for mocking/testing from the new main
from main import (
    app,
    detect_language,
    extract_code_block,
    chunk_code_by_lines,  # Renamed function
    load_prompts,
    call_llm_api,  # New function using requests
    call_llm_with_retry,  # New retry wrapper
    improve_code_chunk,
    generate_commit_message,
    process_code,
    LLMAPIError,  # New custom exception
    DEFAULT_COMMIT_MESSAGE,
    IMPROVEMENTS_DIR,  # Use the Path object from main
    BASE_URL,  # Import config for testing route logic
    LLM_API_TYPE
)


# Mock the logger to prevent log output during tests unless needed
@patch('main.logger', MagicMock())
class TestSciFiImproved(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up once for all tests."""
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'  # Use a fixed key for testing sessions
        # Disable CSRF protection if you were using Flask-WTF (not present here, but good practice)
        # app.config['WTF_CSRF_ENABLED'] = False

        # Create test prompts file
        cls.test_prompts_content = {
            "improve_code": "Test improve prompt: lang={language}, chunk#={chunk_number}/{total_chunks}, hint={context_hint}\n```\n{code_chunk}\n```",
            "commit_message": "Test commit prompt: lang={language}\nCode:\n{improved_code}"
        }
        cls.prompts_filename = "test_prompts.json"
        with open(cls.prompts_filename, "w", encoding='utf-8') as f:
            json.dump(cls.test_prompts_content, f)

        # Patch load_prompts to use our test file during tests
        # Note: using load_prompts() directly here loads it *once* when the class is defined.
        # If main.py changed prompts.json *after* this line, the test wouldn't see it.
        # Fine for this setup, but for dynamic loading, mocking might need adjustment.
        cls.loaded_test_prompts = load_prompts(cls.prompts_filename)
        cls.load_prompts_patcher = patch('main.load_prompts', return_value=cls.loaded_test_prompts)
        cls.load_prompts_patcher.start()

        # Ensure improvements directory exists for cleanup
        IMPROVEMENTS_DIR.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up once after all tests."""
        cls.load_prompts_patcher.stop()
        if os.path.exists(cls.prompts_filename):
            os.remove(cls.prompts_filename)
        if IMPROVEMENTS_DIR.exists():
            try:
                shutil.rmtree(IMPROVEMENTS_DIR)
            except OSError as e:
                print(f"Error removing directory {IMPROVEMENTS_DIR}: {e}", file=sys.stderr)


    def setUp(self):
        """Set up before each test method."""
        self.client = app.test_client()
        # Clear session before each test
        with self.client.session_transaction() as sess:
            sess.clear()

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any files created *within* a test if necessary
        # Example: ensure improvements dir is clean if a test adds specific files
        for item in IMPROVEMENTS_DIR.glob('*.json'):
             try:
                 os.remove(item)
             except OSError as e:
                 print(f"Error removing file {item}: {e}", file=sys.stderr)
        pass

    # --- Test Helper Functions ---

    def test_detect_language(self):
        python_code = "import os\ndef test():\n    pass"
        js_code = "function test() { return true; }"
        java_code = "public class Test { public static void main(String[] args){ System.out.println(); } }" # Added ;
        go_code = "package main\nimport \"fmt\"\nfunc main() {}"
        c_code = "#include <stdio.h>\nint main() { return 0; }"
        html_code = "<html><body><div>Test</div></body></html>"
        css_code = ".class { color: red; }"
        php_code = "<?php echo 'Hello'; ?>"

        self.assertEqual(detect_language(python_code), "python")
        self.assertEqual(detect_language(js_code), "javascript")
        self.assertEqual(detect_language(java_code), "java")
        self.assertEqual(detect_language(go_code), "go")
        self.assertEqual(detect_language(c_code), "c")
        self.assertEqual(detect_language(html_code), "html")
        self.assertEqual(detect_language(css_code), "css")
        self.assertEqual(detect_language(php_code), "php")
        self.assertEqual(detect_language("random text without keywords"), "python") # Default
        self.assertEqual(detect_language(None), "python") # Handle None input
        self.assertEqual(detect_language(123), "python")  # Handle non-string input

    def test_extract_code_block(self):
        # Assuming main.py's extract_code_block was modified to be lenient as discussed
        text_with_lang = "Some text before\n```python\nprint('hello')\n# comment\n```\nSome text after"
        self.assertEqual(extract_code_block(text_with_lang, "python"), "print('hello')\n# comment")

        text_no_lang = "```\nprint('world')\n```"
        self.assertEqual(extract_code_block(text_no_lang), "print('world')")
        # Should pass now if main.py was changed
        self.assertEqual(extract_code_block(text_no_lang, "python"), "print('world')")

        text_mismatch_lang = "```javascript\nconsole.log('hi');\n```"
        # Should pass now if main.py was changed, returning the JS content
        self.assertEqual(extract_code_block(text_mismatch_lang, "python"), "console.log('hi');")

        plain_text = "just plain code"
        self.assertEqual(extract_code_block(plain_text), "just plain code")
        self.assertEqual(extract_code_block(plain_text, "python"), "just plain code")

        empty_block = "```python\n```"
        self.assertEqual(extract_code_block(empty_block, "python"), "")

        not_a_block = "```python print('hi') ```"
        self.assertEqual(extract_code_block(not_a_block, "python"), not_a_block)

    def test_chunk_code_by_lines(self):
        code = "line1\nline2\nline3\nline4\nline5"
        chunks = chunk_code_by_lines(code, max_lines=2)
        self.assertEqual(len(chunks), 3)
        self.assertIsInstance(chunks, list)
        self.assertIsInstance(chunks[0], str)
        self.assertEqual(chunks[0], "line1\nline2")
        self.assertEqual(chunks[1], "line3\nline4")
        self.assertEqual(chunks[2], "line5")

        empty_code = ""
        self.assertEqual(chunk_code_by_lines(empty_code, 10), [])

        single_line = "one line"
        self.assertEqual(chunk_code_by_lines(single_line, 10), ["one line"])

    # --- Test API Calling Logic ---

    @patch('main.requests.post')
    def test_call_llm_api_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " test response "}}]
        }
        mock_post.return_value = mock_response

        result = call_llm_api("test prompt", "test_key", "http://test.com", "test_model", 100, 0.5, "openrouter")

        self.assertEqual(result, "test response")
        expected_url = "http://test.com/chat/completions"
        expected_headers = {"Content-Type": "application/json", "Authorization": "Bearer test_key"}
        expected_data = {"model": "test_model", "messages": [{"role": "user", "content": "test prompt"}], "max_tokens": 100, "temperature": 0.5}
        mock_post.assert_called_once_with(expected_url, headers=expected_headers, json=expected_data, timeout=120)

    @patch('main.requests.post')
    def test_call_llm_api_success_openai_type(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "openai response"}}]}
        mock_post.return_value = mock_response

        result = call_llm_api("prompt", "key", "http://localhost:1234", "model", 50, 0.1, "openai")
        self.assertEqual(result, "openai response")
        expected_url = "http://localhost:1234/v1/chat/completions"
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], expected_url)

    @patch('main.requests.post')
    def test_call_llm_api_success_no_key_local(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "local response"}}]}
        mock_post.return_value = mock_response

        result = call_llm_api("prompt", "", "http://localhost:1234", "local-model", 50, 0.1, "openai")
        self.assertEqual(result, "local response")
        expected_url = "http://localhost:1234/v1/chat/completions"
        expected_headers = {"Content-Type": "application/json"}
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], expected_url)
        self.assertEqual(mock_post.call_args[1]['headers'], expected_headers)


    @patch('main.requests.post')
    def test_call_llm_api_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response) # Use imported requests
        mock_post.return_value = mock_response

        with self.assertRaises(LLMAPIError) as cm:
            call_llm_api("p", "k", "url", "m", 1, 0, "openai")
        self.assertIn("Network error", str(cm.exception))
        self.assertEqual(cm.exception.status_code, 500)

    @patch('main.requests.post')
    def test_call_llm_api_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Cannot connect") # Use imported requests

        with self.assertRaises(LLMAPIError) as cm:
            call_llm_api("p", "k", "url", "m", 1, 0, "openai")
        self.assertIn("Network error contacting LLM API: Cannot connect", str(cm.exception))
        self.assertIsNone(cm.exception.status_code)

    @patch('main.requests.post')
    def test_call_llm_api_json_decode_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "This is not JSON"
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "This is not JSON", 0)
        mock_post.return_value = mock_response

        with self.assertRaises(LLMAPIError) as cm:
            call_llm_api("p", "k", "url", "m", 1, 0, "openai")
        self.assertIn("Invalid JSON response", str(cm.exception))

    @patch('main.requests.post')
    def test_call_llm_api_internal_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": {"message": "Invalid API key", "code": "auth_error"}}
        mock_post.return_value = mock_response

        with self.assertRaises(LLMAPIError) as cm:
            call_llm_api("p", "k", "url", "m", 1, 0, "openai")
        self.assertIn("API Error: Invalid API key", str(cm.exception))

    @patch('main.requests.post')
    def test_call_llm_api_unexpected_structure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "some data"} # Missing 'choices'
        mock_post.return_value = mock_response

        with self.assertRaises(LLMAPIError) as cm:
            call_llm_api("p", "k", "url", "m", 1, 0, "openai")
        self.assertIn("Unexpected API response structure", str(cm.exception))

    # --- Test Retry Logic ---

    @patch('main.time.sleep', return_value=None)
    @patch('main.call_llm_api')
    def test_call_llm_with_retry_success_first_try(self, mock_call_llm_api, mock_sleep):
        mock_call_llm_api.return_value = "Success!"
        prompt_gen = MagicMock(return_value="Generated Prompt")

        result = call_llm_with_retry(prompt_gen, "api_key", "arg1", kwarg1="kw1")

        self.assertEqual(result, "Success!")
        prompt_gen.assert_called_once_with("arg1", kwarg1="kw1")
        mock_call_llm_api.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('main.time.sleep', return_value=None)
    @patch('main.call_llm_api')
    def test_call_llm_with_retry_success_after_failures(self, mock_call_llm_api, mock_sleep):
        mock_call_llm_api.side_effect = [LLMAPIError("Fail1"), LLMAPIError("Fail2"), "Success!"]
        prompt_gen = MagicMock(return_value="Generated Prompt")

        with patch('main.MAX_RETRIES', 2):
             result = call_llm_with_retry(prompt_gen, "api_key", "arg1")

        self.assertEqual(result, "Success!")
        self.assertEqual(prompt_gen.call_count, 3)
        self.assertEqual(mock_call_llm_api.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('main.time.sleep', return_value=None)
    @patch('main.call_llm_api')
    def test_call_llm_with_retry_all_failures(self, mock_call_llm_api, mock_sleep):
        final_error = LLMAPIError("Final Fail")
        mock_call_llm_api.side_effect = [LLMAPIError("Fail1"), LLMAPIError("Fail2"), final_error]
        prompt_gen = MagicMock(return_value="Generated Prompt")

        with patch('main.MAX_RETRIES', 2):
            with self.assertRaises(LLMAPIError) as cm:
                 call_llm_with_retry(prompt_gen, "api_key")

        self.assertIs(cm.exception, final_error)
        self.assertEqual(prompt_gen.call_count, 3)
        self.assertEqual(mock_call_llm_api.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    # --- Test Core Logic Functions ---

    @patch('main.call_llm_with_retry')
    def test_improve_code_chunk_success(self, mock_retry):
        mock_retry.return_value = "```python\nimproved chunk\n```"
        original_chunk = "original chunk"

        result = improve_code_chunk(original_chunk, "python", 1, 2, "api_key", "hint")

        self.assertEqual(result, "improved chunk")
        mock_retry.assert_called_once()
        args, kwargs = mock_retry.call_args
        self.assertEqual(args[1], "api_key")
        prompt_func_args = args[2:]
        self.assertEqual(prompt_func_args, ("python", 1, 2, "hint", original_chunk))

    @patch('main.call_llm_with_retry', side_effect=LLMAPIError("Failed"))
    def test_improve_code_chunk_failure(self, mock_retry):
        original_chunk = "original chunk"
        result = improve_code_chunk(original_chunk, "python", 1, 1, "api_key")
        self.assertEqual(result, original_chunk)
        mock_retry.assert_called_once()

    @patch('main.call_llm_with_retry')
    def test_generate_commit_message_success(self, mock_retry):
        mock_retry.return_value = "feat: Generated commit message"
        # Ensure string is long enough (> 2000 chars) for truncation
        improved_code = ("some improved code " * 150).strip() # Approx 2700 chars

        result = generate_commit_message(improved_code, "python", "api_key")

        self.assertEqual(result, "feat: Generated commit message")
        mock_retry.assert_called_once()
        args, kwargs = mock_retry.call_args
        self.assertEqual(args[1], "api_key")
        prompt_func_args = args[2:]
        self.assertEqual(prompt_func_args[0], "python")
        passed_code = prompt_func_args[1]
        self.assertTrue(passed_code.startswith("some improved code"))
        # Assert it was actually truncated and ends with the suffix
        self.assertTrue(passed_code.endswith("... (code truncated for brevity)"),
                        f"String did not end with suffix. Actual end: '{passed_code[-50:]}'")
        self.assertLess(len(passed_code), len(improved_code)) # Verify length decreased

    @patch('main.call_llm_with_retry', side_effect=LLMAPIError("Failed"))
    def test_generate_commit_message_failure(self, mock_retry):
        result = generate_commit_message("code", "python", "api_key")
        self.assertEqual(result, DEFAULT_COMMIT_MESSAGE)
        mock_retry.assert_called_once()

    @patch('main.generate_commit_message')
    @patch('main.improve_code_chunk')
    def test_process_code_success(self, mock_improve_chunk, mock_gen_commit):
        original_code = "chunk1 line1\nchunk1 line2\nchunk2 line1"
        # Simulate outputs for each chunk call
        mock_improve_chunk.side_effect = ["improved chunk1", "improved chunk2"]
        mock_gen_commit.return_value = "Test commit message"
        expected_improved_code = "improved chunk1\nimproved chunk2"

        with patch('main.CHUNK_SIZE_LINES', 2):
             improved_code, commit_msg, error = process_code(original_code, "python", "api_key")

        self.assertIsNone(error)
        self.assertEqual(improved_code, expected_improved_code)
        self.assertEqual(commit_msg, "Test commit message")

        self.assertEqual(mock_improve_chunk.call_count, 2)
        # Correct the expected context hint for the second call based on main.py's logic
        calls = [
             call("chunk1 line1\nchunk1 line2", "python", 1, 2, "api_key", context_hint=""),
             call("chunk2 line1", "python", 2, 2, "api_key", context_hint="improved chunk1") # Correct hint
        ]
        mock_improve_chunk.assert_has_calls(calls, any_order=False)

        mock_gen_commit.assert_called_once_with(expected_improved_code, "python", "api_key")

    @patch('main.improve_code_chunk', side_effect=LLMAPIError("Chunk failed"))
    def test_process_code_chunk_failure(self, mock_improve_chunk):
        original_code = "line1\nline2"
        improved_code, commit_msg, error = process_code(original_code, "python", "api_key")

        self.assertIsNotNone(error)
        self.assertIn("LLM API Error during processing: Chunk failed", error)
        self.assertEqual(improved_code, "")
        self.assertEqual(commit_msg, "")
        mock_improve_chunk.assert_called_once()

    @patch('main.improve_code_chunk', return_value="improved code")
    @patch('main.generate_commit_message', side_effect=LLMAPIError("Commit failed"))
    def test_process_code_commit_failure(self, mock_gen_commit, mock_improve_chunk):
        original_code = "line1"
        improved_code, commit_msg, error = process_code(original_code, "python", "api_key")

        self.assertIsNotNone(error)
        self.assertIn("LLM API Error during processing: Commit failed", error)
        self.assertEqual(improved_code, "")
        self.assertEqual(commit_msg, "")
        mock_improve_chunk.assert_called_once()
        mock_gen_commit.assert_called_once()

    # --- Test Flask Routes ---

    @patch('main.process_code')
    def test_improve_route_success(self, mock_process_code):
        mock_process_code.return_value = ("improved code result", "commit message result", None)
        test_code = "def test(): pass"
        test_api_key = "test-key-123"

        response = self.client.post('/improve', json={'code': test_code, 'language': 'python', 'api_key': test_api_key})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("improved_code", data)
        self.assertIn("commit_message", data)
        self.assertIn("iteration", data)
        self.assertEqual(data['improved_code'], "improved code result")
        self.assertTrue(data['commit_message'].startswith("commit message result"))
        self.assertIn("Improvement generated by", data['commit_message'])
        self.assertEqual(data['iteration'], 1)

        # Basic check file was saved
        saved_files = list(IMPROVEMENTS_DIR.glob('improvement_*.json'))
        self.assertGreater(len(saved_files), 0, "No improvement file was saved")
        # Optional: Check content of saved file
        # with open(saved_files[0], 'r') as f:
        #     saved_data = json.load(f)
        #     self.assertEqual(saved_data['improved_code'], "improved code result")


        with self.client.session_transaction() as sess:
            self.assertIn("version_history", sess)
            self.assertEqual(len(sess["version_history"]), 1)
            self.assertEqual(sess["version_history"][0], "improved code result")

        mock_process_code.assert_called_once()
        args, kwargs = mock_process_code.call_args
        # Assuming bleach doesn't significantly alter this simple code
        self.assertEqual(args[0], test_code)
        self.assertEqual(args[1], 'python')
        self.assertEqual(args[2], test_api_key)

    @patch('main.process_code', return_value=("", "", "Simulated LLM API Error: Bad Key"))
    def test_improve_route_api_error(self, mock_process_code):
        response = self.client.post('/improve', json={'code': 'test', 'language': 'python', 'api_key': 'key'})
        self.assertEqual(response.status_code, 400) # API errors -> 400
        self.assertIn("error", response.get_json())
        self.assertEqual(response.get_json()['error'], "Simulated LLM API Error: Bad Key")

    @patch('main.process_code', side_effect=Exception("Unexpected internal error processing"))
    def test_improve_route_internal_error(self, mock_process_code):
        response = self.client.post('/improve', json={'code': 'test', 'language': 'python', 'api_key': 'key'})
        self.assertEqual(response.status_code, 500) # Internal errors -> 500
        self.assertIn("error", response.get_json())
        self.assertIn("Unexpected internal error", response.get_json()['error']) # Check substring

    def test_improve_route_no_code(self):
        response = self.client.post('/improve', json={'language': 'python', 'api_key': 'key'})
        self.assertEqual(response.status_code, 400)
        self.assertIn("No code provided", response.get_json()['error'])

    # Simulate remote URL where key is typically required by API
    @patch('main.BASE_URL', "https://api.remote.com")
    @patch('main.LLM_API_TYPE', "openrouter")
    def test_improve_route_no_key_remote_required(self, mock_api_type, mock_base_url):
        # Test the flow where the API key is missing but might be needed.
        # The route itself allows it, but process_code should fail if call_llm_api needs it.
        with patch('main.process_code', return_value=("", "", "LLM API Error: Missing Key (Simulated)")) as mock_proc:
            response = self.client.post('/improve', json={'code': 'test', 'language': 'python'}) # No api_key
            self.assertEqual(response.status_code, 400) # Expect failure reported by process_code
            self.assertIn("LLM API Error: Missing Key", response.get_json()['error'])
            mock_proc.assert_called_once_with('test', 'python', '') # process_code called with empty key

    # Add parameters to accept mocks from @patch decorators (order is reversed)
    @patch('main.process_code') # Inner decorator -> last argument
    @patch('main.LLM_API_TYPE', "openai") # Middle decorator -> middle argument
    @patch('main.BASE_URL', "http://localhost:11434") # Outer decorator -> first argument after self
    def test_improve_route_no_key_local(self, mock_base_url_obj, mock_api_type_obj, mock_process_code):
        """ Test route when API key is omitted for a local URL (should succeed if API doesn't require key). """
        mock_process_code.return_value = ("local success", "local commit", None)

        response = self.client.post('/improve', json={'code': 'test', 'language': 'python'}) # No api_key

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['improved_code'], "local success")
        # Verify process_code was called with an empty API key string
        mock_process_code.assert_called_once_with('test', 'python', '')

    def test_detect_language_route(self):
        response = self.client.post('/detect_language', json={'code': 'function hello() {}'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['language'], 'javascript')
        self.assertTrue(data['detected'])

    def test_detect_language_route_no_code(self):
        response = self.client.post('/detect_language', json={})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['language'], 'python') # Defaults
        self.assertFalse(data['detected'])

    def test_clear_session(self):
        with self.client.session_transaction() as sess:
            sess['version_history'] = ['item1'] # Set something in session

        response = self.client.post('/clear_session')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'Session cleared successfully.')

        # Verify session is now empty
        with self.client.session_transaction() as sess:
            self.assertNotIn('version_history', sess)

    def test_stream_route_exists(self):
         response = self.client.get('/stream')
         self.assertEqual(response.status_code, 200)
         self.assertEqual(response.mimetype, 'text/event-stream')
         # Fully testing SSE content requires a more complex setup (e.g., client library)


if __name__ == '__main__':
    unittest.main()