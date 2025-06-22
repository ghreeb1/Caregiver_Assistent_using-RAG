import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# >>> SOLUTION: We now use a more advanced chain structure that requires these components
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import DB_FAISS_PATH, OLLAMA_MODEL, EMBEDDING_MODEL
# Make sure your prompts.py has the new, concise prompt
from prompts import SYSTEM_PROMPT


def run_chatbot():
    print("Initializing Caregiver Chatbot...")

    # Load embeddings model
    print("Loading embedding model...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL,
                                          model_kwargs={'device': 'cpu'},  # Use 'cuda' if available
                                          encode_kwargs={'normalize_embeddings': True})

    # Load FAISS vector store
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector database not found at {DB_FAISS_PATH}.")
        print("Please run 'python ingest.py' first to create the database.")
        return

    print(f"Loading vector database from {DB_FAISS_PATH}...")
    vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    print("Vector database loaded. Retriever will fetch top 4 documents.")

    # Initialize Ollama LLM
    print(f"Initializing Ollama model: {OLLAMA_MODEL}...")
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2) # Lower temperature for more factual, less creative answers
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is downloaded.")
        return
    print("Ollama model initialized.")


    # >>> SOLUTION: REVISED AND MORE EFFECTIVE RAG CHAIN FOR CONCISE ANSWERS

    print("Creating RAG chain for concise answers...")

    # This is the prompt template that uses the concise instructions from prompts.py
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # This chain will generate the final, brief answer based on the (now filtered) context.
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    # We now create the full retrieval chain which combines the retriever and the document_chain
    # The retriever will fetch documents, and they will be "stuffed" into the document_chain's prompt
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG chain created.")


    print("\n--- Caregiver Chatbot Ready ---")
    print("Type your questions about caregiving. Type 'exit' or 'quit' to end the chat.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Take care.")
            break

        if not user_input:
            continue

        try:
            print("\nChatbot: ", end="", flush=True)

            # We use `stream` for a responsive feel
            full_answer = ""
            for chunk in retrieval_chain.stream({"input": user_input}):
                if answer_part := chunk.get("answer"):
                    print(answer_part, end="", flush=True)
                    full_answer += answer_part
            print()

        except Exception as e:
            print(f"\nChatbot: An error occurred: {e}")
            print("Please check your Ollama server connection or the model availability.")


if __name__ == "__main__":
    run_chatbot()