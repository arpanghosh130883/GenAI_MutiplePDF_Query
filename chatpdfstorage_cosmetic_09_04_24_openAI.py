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
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Load environment variables and configure the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# genai.configure(api_key="XXXXX")  ### Uncomment this when we are deploying in Streamlit cloud Configuring the API key for the streamlit app to work in streamlit cloud after deployment

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
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception as e:
                print(f"Error extracting text from page: {e}")
    return text

def list_stored_pdfs():
    """Lists PDF files stored in the PDF directory."""
    return [f for f in os.listdir(PDF_DIR) if os.path.isfile(os.path.join(PDF_DIR, f))]

def get_text_chunks(text):
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a vector store from text chunks."""
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Creates and returns a conversational chain for the chatbot."""
    prompt_template = """
    Act as AI-PDF expert. Users upload one or more PDF files and ask you questions based on those uploaded files.
    Your job is to understand the question and generate as detailed as possible answers based on the context of PDF. 
    Identify one or more paragraphs that contain relevant information and combine them to provide a long and detailed answer.
    If the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
   # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    #model = genai.GenerativeModel('gemini-1.5-flash') #  If you are getting quota limit for pro modle use this model
    model = ChatOpenAI(
    model="gpt-4",         # or "gpt-3.5-turbo"
    temperature=0.3,
    #api_key="your-openai-api-key"  # or set it via env variable
)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Processes the user's question and displays the response."""
    detailed_question = user_question + " Explain in detail."
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(detailed_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": detailed_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        if "stop condition" in str(e):
            st.write("Reply:", "Sorry, I couldn't find an answer based on the provided context.")
        else:
            raise e

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config("Chat PDF")

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
        # Set max upload size to 200MB per file (adjust as needed)
        #st.set_option("server.maxUploadSize", 200)
        uploaded_files = st.file_uploader("Upload PDF files here", type="pdf", accept_multiple_files=True)
        if st.button("Process Uploaded PDFs"):
            with st.spinner("Processing..."):
                pdf_paths = save_uploaded_files(uploaded_files)
                raw_text = get_pdf_text(pdf_paths)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")
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
