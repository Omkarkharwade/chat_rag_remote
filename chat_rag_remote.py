import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="C++ RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 C++ RAG Chatbot using Ollama + Mistral")


# -----------------------------
# Load & Process Data
# -----------------------------
@st.cache_resource
def load_vectorstore():

    # Load C++ text file
    loader = TextLoader(
        "C++_Introduction.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    final_docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    db = FAISS.from_documents(
        final_docs,
        embeddings
    )

    return db


# Load vector database
db = load_vectorstore()


# -----------------------------
# Load Mistral from Ollama
# -----------------------------
llm = OllamaLLM(
    model="mistral"
)


# -----------------------------
# Chat Interface
# -----------------------------
user_question = st.text_input(
    "Ask a question about C++:"
)


# -----------------------------
# Generate Answer
# -----------------------------
if user_question:

    with st.spinner("Thinking..."):

        # Search relevant documents
        docs = db.similarity_search(
            user_question,
            k=4
        )

        # Extract context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Create prompt
        prompt = f"""
You are a helpful C++ tutor.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Context:
--------------------
{context}
--------------------

Question:
{user_question}

Answer:
"""

        # Get response from Mistral
        response = llm.invoke(prompt)

        # Display answer
        st.subheader("🤖 Answer")

        st.write(response)


        # -----------------------------
        # Show Sources
        # -----------------------------
        with st.expander("📚 View Retrieved Context"):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"**Source {i + 1}:**"
                )

                st.write(
                    doc.page_content
                )