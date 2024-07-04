import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
# import FAISS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chat with Your PDFs", layout="wide")

# Load environment variables and configure the API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=api_key)

# Directory to store uploaded PDFs
PDF_DIR = "uploaded_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

def save_uploaded_files(uploaded_files):
    """Saves uploaded files to the server and returns their paths."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def get_pdf_text(pdf_paths):
    """Extracts text from the given PDF paths."""
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.warning(f"No text found on page {page_num} of {pdf_path}")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
    return text

def list_stored_pdfs():
    """Lists PDF files stored in the PDF directory."""
    return [f for f in os.listdir(PDF_DIR) if os.path.isfile(os.path.join(PDF_DIR, f))]

def get_text_chunks(text):
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

FAISS_INDEX_PATH = "faiss_index"

def get_vector_store(text_chunks):
    """Creates and saves a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    st.success("FAISS index created and saved successfully!")

# Function to load FAISS index
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.success("Embeddings loaded successfully!")
    faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.success("FAISS index loaded successfully!")
    return faiss_index
    

def get_conversational_chain():
    """Creates and returns a conversational chain for the chatbot."""
    prompt_template = """
    Act as an AI-PDF expert. Users upload one or more PDF files and ask you questions based on those uploaded files.
    Your job is to understand the question and generate as detailed as possible answers based on the context of the PDF. 
    Identify one or more paragraphs that contain relevant information and combine them to provide a long and detailed answer.
    If the answer is not in the provided context just say, "The answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Processes the user's question and displays the response."""
    detailed_question = user_question + " Explain in detail."
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error("FAISS index file not found. Please ensure the file exists.")
            return
        new_db = load_faiss_index()

        if new_db is None:
            st.error("Failed to load FAISS index.")
            return
        
        docs = new_db.similarity_search(detailed_question)
        if not docs:
            st.write("No similar documents found.")
            return
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": detailed_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        st.error("FAISS index file not found. Please ensure the file exists.")
    except ValueError as ve:
        logger.error(f"ValueError while loading FAISS index: {ve}")
        st.error("ValueError while loading FAISS index. Please check the embeddings and index file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        st.error("An unexpected error occurred. Please check the logs for more details.")

def main():
    """Main function to run the Streamlit app."""
    # st.set_page_config(page_title="Chat with Your PDFs", layout="wide")

    # Custom CSS for black background
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #333;
            color: white;
        }
        .stButton>button {
            border: 2px solid #4CAF50;
            background-color: #333;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Chat with Your PDFs")

    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDF files here", type="pdf", accept_multiple_files=True)
        if st.button("Process Uploaded PDFs"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    pdf_paths = save_uploaded_files(uploaded_files)
                    raw_text = get_pdf_text(pdf_paths)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("No text extracted from the PDFs.")
            else:
                st.error("No PDF files uploaded. Please upload PDF files to process.")
        
        # Display the list of stored PDFs
        st.header("Stored PDFs")
        for pdf_file in list_stored_pdfs():
            st.text(pdf_file)

    st.header("Ask a Question")
    user_question = st.text_input("Enter your question here:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
