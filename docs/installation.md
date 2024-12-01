## Prerequisites

Before installing the application, ensure you have:

- Python 3.8 or higher installed
- [Ollama](https://ollama.ai/) installed and running locally
- Sufficient disk space for vector storage

## Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/f-zh-oubella/RAG_app.git>
   cd RAG_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama service:
   ```bash
   ollama serve
   ```

5. Pull the required model:
   ```bash
   ollama pull llama3.1
   ```

## Verifying Installation

To verify that everything is installed correctly:

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Try uploading a PDF and asking a question
