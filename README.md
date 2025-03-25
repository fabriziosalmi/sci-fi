# Simple Code Improver - Framework & Interface

## Overview
Sci-Fi Code Improver is a web application that enhances your programming code by analyzing and improving it for security, performance, maintainability, and adherence to coding best practices. It also generates commit messages following the Conventional Commits format.

## Features
- **Code Improvement:** Automatically refactors code while preserving functionality.
- **Commit Message Generator:** Creates informative commit messages conforming to Conventional Commits.
- **Syntax Highlighting:** Supports multiple programming languages.
- **Theme Toggle:** Switch between light and dark themes.
- **Session Management:** Version history maintained in user sessions.
- **Auto Language Detection:** Determines the coding language from the input.
- **Docker Support:** Easy deployment using Docker containers.
- **API Flexibility:** Support for both OpenRouter and OpenAI APIs.
- **Health Monitoring:** Built-in healthcheck and logging system.
- **FreeTekno modding:** [Background video FX](https://pixabay.com/it/users/ceos_stock-13890949/) and [Free Undeground Tekno Radio](https://radio.free-tekno.com) music for a unique experience!

## Screenshot

![screenshot](https://github.com/fabriziosalmi/sci-fi/blob/main/static/sci-fi_screenshot.png?raw=true)

## Quick Start with Docker

```bash
# Pull the Docker image
docker pull fabriziosalmi/sci-fi:latest

# Create a .env file with your configuration
cat > .env << EOL
OPENROUTER_API_KEY=your-api-key
MODEL_NAME=qwen2.5-coder-3b-instruct-mlx
MAX_TOKENS=2048
TEMPERATURE=0.2
CHUNK_SIZE=500
CONTEXT_WINDOW=10
BASE_URL=http://localhost:1234/v1
EOL

# Run the container
docker run -p 7860:7860 --env-file .env fabriziosalmi/sci-fi:latest
```

## Traditional Installation
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
   Create a `.env` file in the project root with your configuration.

## Docker Compose Setup
1. **Create docker-compose.yml:**
   ```yaml
   version: '3.8'
   services:
     web:
       image: fabriziosalmi/sci-fi:latest
       ports:
         - "7860:7860"
       volumes:
         - ./improvements:/app/improvements
         - ./.env:/app/.env:ro
       environment:
         - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
         - MODEL_NAME=${MODEL_NAME}
         - MAX_TOKENS=${MAX_TOKENS}
         - TEMPERATURE=${TEMPERATURE}
         - CHUNK_SIZE=${CHUNK_SIZE}
         - CONTEXT_WINDOW=${CONTEXT_WINDOW}
         - BASE_URL=${BASE_URL}
       restart: unless-stopped
   ```

2. **Start the application:**
   ```bash
   docker-compose up -d
   ```

## Usage
1. Access the app at [http://localhost:7860](http://localhost:7860)
2. Paste your code into the text area
3. Click the magic icon to improve the code
4. View the improved code and generated commit message
5. Use the copy/download buttons to handle output

## API Configuration
The application supports both OpenRouter and OpenAI APIs:
- **OpenRouter:** Set `BASE_URL=https://openrouter.ai/api/v1`
- **OpenAI:** Set `BASE_URL=https://api.openai.com/v1` and `LLM_API_TYPE=openai`

## Health Monitoring
The Docker container includes:
- Health checks every 30 seconds
- JSON file logging with rotation to efficiently manage log data.
- Maximum log file size of 10MB to prevent excessive disk usage.
- Retains the 3 most recent log files for quick audits.

## Contributing
Contributions are welcome! Please:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with details of your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements
- Docker support for streamlined deployment.
