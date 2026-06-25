# Enterprise Knowledge Assistant
This is an implementation of an assistant that utilises Langchain agents to answer user questions, 
using a knowledge base formed from input documents that has been converted into a vector store.
It has a Streamlit front end.
- [Installation](#installation)
- [Usage](#usage)
  - [Vector store creation](#vector-store-creation)
  - [Streamlit app](#streamlit-app)
- [Tests](#tests)

## Installation
Clone the repo:
```
git clone https://github.com/jambent/enterprise_knowledge_assistant.git
```
Create a Python virtual environment.  
For example, using venv:
```
python -m venv venv
```
Activate that virtual environment.  On Linux or Mac this would require
```
source venv/bin/activate
```
but on Windows:
```
cd venv/Scripts
./activate.ps1
```
If you have changed directory, ensure you return to the root directory
of the project.
Then install dependencies from the requirements.txt file:
```
pip install -r requirements.txt
```
Finally, you will need to create a .env file in order to hold your LLM endpoint.
This should hold a single AGENT_URL value, only
```
AGENT_URL=<LLM_ENDPOINT>
```

## Usage
### Vector store creation
First a vector store has to be created from input files.  
Note that this implementation will only work with .pdf, .txt
and .docx input files.
Create a directory called input_files at the root of the project
```
mkdir input_files
```
Then place the desired files for your knowledge base within 
subdirectories within this input_files directory, e.g., input_files/personnel/file.pdf.
NOTE: This implementation does not allow files to be placed in the input_files directory itself.
.

When ready to create the vector store run the following:
```
python -m src.create_vectorstore_from_enterprise_docs
```
When file processing starts you will see an indication of which
files are being processed and which have been converted to Markdown.
You will also see that a knowledge_base directory has been created 
to hold the Markdown files.
Once complete, a preprocessed_db directory will also be present.
This contains the Chroma database that holds the created vectors.
#### Vector store artifacts
If desired you can retrieve the documents from the vector store, including document content, metadata
and the vectors themselves.
Run
```
python -m src.utilities.vectorstore_artifact_retrieval
```
A vectorstore_artifacts directory will be created, which will contain a .jsonl file containing this information. 

### Streamlit app
Once the vector store has been created, you are ready to use the knowledge assistant.
Simply run the following to start the associated Streamlit app:
```
streamlit run app.py
```
This should automatically open the assistant at the Local URL indicated in the terminal.
Enter questions in the bar at the bottom of the screen and the assistant will respond above, the history of the 
chat session scrolling upwards automatically with each question.
Tailor the app.py code and the system prompt for the reasoning agent (src/assistant_agents/reasoning) to reflect the particular company that the assistant is required to
answer questions about for you.
#### Logs
A logs directory will be created, containing app.log files.
These capture the user and assistant interactions, along with the user identity and session id.

### Tests
The associated test suite can be run by
```
python -m pytest
```

