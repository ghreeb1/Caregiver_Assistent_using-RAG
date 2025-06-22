How to Create rag-app Environment with Anaconda
Follow these steps to set up a Python environment for the RAG (Retrieval-Augmented Generation) application using Anaconda.

🔹 Step 1: Open Anaconda Prompt
Click on Start.

Search for Anaconda Prompt.

Click to open it.

This opens your command-line interface for working with Conda.

🔹 Step 2: Create a New Environment
Run the following command to create a new environment named rag-app with Python 3.10:

conda create --name rag-app python=3.10 -y
--name rag-app: sets the name of the environment

python=3.10: specifies the Python version

-y: automatically confirms the installation

🔹 Step 3: Activate the Environment
Activate the new environment by running:

conda activate rag-app
You should see the prompt change to something like:

(rag-app) C:\Users\YourName>
This indicates that you are now inside the rag-app environment.

🔹 Step 4: Install RAG Project Dependencies
Install the required packages using pip:

pip install fastapi uvicorn langchain langchain-core langchain-community langchain_ollama faiss-cpu python-dotenv sentence-transformers openai unstructured pypdf pdfminer.six
