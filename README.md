# Sci-Fi Code Improver

## Overview
Sci-Fi Code Improver is a web application that enhances your programming code by analyzing and improving it for security, performance, maintainability, and adherence to coding best practices. It also generates commit messages following the Conventional Commits format.

## Features
- **Code Improvement:** Automatically refactors code while preserving functionality.
- **Commit Message Generator:** Creates informative commit messages conforming to Conventional Commits.
- **Syntax Highlighting:** Supports multiple programming languages.
- **Theme Toggle:** Switch between light and dark themes.
- **Session Management:** Version history maintained in user sessions.
- **Auto Language Detection:** Determines the coding language from the input.
- **FreeTekno addons:** Background tripping video and [Free Undeground Tekno Radio](https://radio.free-tekno.com) music 🌀

## Screenshot

![screenshot](https://github.com/fabriziosalmi/sci-fi/blob/main/static/sci-fi_screenshot.png?raw=true)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/fabriziosalmi/sci-fi.git
   cd sci-fi
   ```
2. **Set up the Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables:**
   Create a `.env` file in the project root like this:
   ```ini
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxx
   MODEL_NAME=google/gemini-2.0-pro-exp-02-05:free
   MAX_TOKENS=2048
   TEMPERATURE=0.2
   CHUNK_SIZE=500
   CONTEXT_WINDOW=10
   BASE_URL=https://openrouter.ai/api/v1
   ```

## Usage
1. **Start the Flask Application:**
   ```bash
   python main.py
   ```
2. **Access the App:**
   Open your browser and navigate to [http://localhost:7860](http://localhost:7860).
3. **How It Works:**
   - Paste your code into the text area.
   - Click the magic icon to improve the code.
   - View the improved code and generated commit message.
   - Use the copy/download buttons to handle output.

## File Structure
- **main.py:** Flask application with routes for code improvement, API endpoints, and session handling.
- **templates/index.html:** The main HTML file that integrates code input, output, and interactive UI elements.
- **static/style.css:** Contains the styling for both light and dark themes along with responsive design adjustments.
- **prompts.json:** Contains the prompts used for code improvement and commit message generation.
- **requirements.txt:** Lists all project dependencies.
- **.gitignore:** Excludes sensitive files and directories from version control.

## API Endpoints
- **/**: Renders the main page.
- **/improve:** Receives code and language, enhances the code using OpenAI, and returns improved code with a commit message.
- **/detect_language:** Auto-detects the programming language from the input code.
- **/clear_session:** Clears the session version history.
- **/stream:** Provides server-sent events for debugging logs.

## Testing
The project includes tests written with pytest. To run tests:
```bash
pytest main.py
```

## Contributing
Contributions are welcome! Please submit issues and pull requests via GitHub:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request detailing your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements
- This project uses OpenAI’s API for code improvement.
- Front-end icons provided by Font Awesome.
- Syntax highlighting by Highlight.js.
