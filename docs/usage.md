# Usage Guide


### Starting the Application

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Start the application:
   ```bash
   streamlit run app.py
   ```

3. Access the web interface at http://localhost:8501

### Uploading Documents

1. Click the "Choose a PDF file" button
2. Select your PDF document
3. Wait for the processing confirmation message

### Asking Questions

You can ask two types of questions:
1. Questions about the uploaded PDF content
2. General knowledge questions

The system will automatically determine the appropriate context for responses.

## Example Usage

![Example Chat](../image.png)

In this example:
1. The first question about reinforcement learning is answered using the model's general knowledge
2. The second question about a specific person is answered based on the uploaded CV


## Vector Storage

The application maintains a persistent vector store:
- Located in `./pdf_chroma_db/`
- Preserves document embeddings between sessions
- Enables quick information retrieval

## Response Streaming

Responses are streamed in real-time:
- See answers as they're generated
- Immediate feedback
- Smooth typing animation

## Best Practices

1. **Document Preparation**:
   - Use text-based PDFs
   - Ensure documents are not password protected
   - Split large documents if needed

2. **Question Formulation**:
   - Be specific when asking about document content
   - Use clear, concise questions
   - Provide context when needed

3. **System Performance**:
   - Process one document at a time
   - Allow processing to complete before asking questions
   - Clear chat history for new topics