# Chat Application with Enhanced Capabilities

## Overview

This project implements a chat application using Python's tkinter for the GUI and integrates with Ollama for natural language processing capabilities. It allows users to interact via text input and receive responses enriched by context from a vault of documents.

## Features

- **User Interface**: Built with tkinter, providing a simple GUI for user interaction.
- **Integration with Ollama**: Utilizes Ollama for generating responses based on input and context from a document vault.
- **Document Vault**: Supports uploading and processing of PDF, text, and JSON files to build a vault of contextually relevant documents.
- **Conversation Logging**: Logs user interactions and system responses for future reference.
- **Query Rewriting**: Enhances user queries by rewriting them based on conversation history and relevant context.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HamzaRadaideh/llamaRag
   ```

2. Ollama Configuration: Set up Ollama API by installing and configuring its requirements.

    - (<https://ollama.com/download>)
    - ollama run llama3
    - ollama pull mxbai-embed-large

3. Dependencies: Ensure Python 3.x is installed. Install required Python packages.

    ```python
    pip install -r requirements.txt
    ```

4. Running the Application:

    ```python
    python main.py
    ```

## Usage

- **Uploading Files**: Use the file upload interface (`Upload PDF`, `Upload Text File`, `Upload JSON File`) to add documents to the vault.
- **Chat Interface**: Enter queries in the chat interface to interact with the application and receive responses.
- **Exiting**: Type "exit" to close the application.

## Contributions

Contributions are welcome! If you find issues or want to enhance features, please fork the repository and submit pull requests.

## Graduation Project Team

*COMPS 💻* Github of team members:

- Ahmad Elwan: <https://github.com/AhmadElwan>
- Hamza Radaideh: <https://github.com/HamzaRadaideh>
- Mohammad Subehi: <https://github.com/mosubehi>
- Khaled Samara: <https://github.com/Khaled270>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
