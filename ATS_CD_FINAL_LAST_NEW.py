from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import re
from sklearn.feature_extraction.text import CountVectorizer
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"


genai.configure(api_key="XXXXX")  ### Configuring the API key for the streamlit app to work in streamlit cloud after deployment


# Define the prompt template
prompt_template = """
Job Description: {job_description}
Resume: {resume_text}
JD Match: {jd_match}

As an ATS scanner and a Technical HR Manager, please provide an analysis of the resume based on the job description with the following details:
- JD Match: {jd_match}%
- Experience: [years]
- Skills Missing: [skills missing keywords],
- Overall Summary: [brief summary]
- Position Match: {position_match}
"""

def get_gemini_response(job_description, resume_text, jd_match, position_match):
    # Format the input text using the prompt template
    input_text = prompt_template.format(job_description=job_description, resume_text=resume_text, jd_match=jd_match, position_match=position_match)
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(input_text)
    return response.text

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_skills(text):
    return set(re.findall(r'\b\w+\b', text.lower()))

def calculate_jd_match(job_description, resume):
    # Vectorizer for job description
    jd_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
    jd_vectorized = jd_vectorizer.fit_transform([job_description])
    jd_skills = set(jd_vectorizer.get_feature_names_out())

    # Vectorizer for resume
    resume_skills = extract_skills(resume)
    #resume_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
    #resume_vectorized = resume_vectorizer.fit_transform([resume])
    #resume_skills = set(resume_vectorizer.get_feature_names_out())

    # Calculate match percentage
    match_percentage = len(jd_skills.intersection(resume_skills)) / len(jd_skills) * 100 if jd_skills else 0
    return round(match_percentage, 2)


def assess_position_match(jd_match, resume_text, job_description_text):
    # Vectorizer for job description
    jd_vectorizer1 = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
    jd_vectorized = jd_vectorizer1.fit_transform([job_description_text])
    jd_skills = set(jd_vectorizer1.get_feature_names_out())

    # Vectorizer for resume
    resume_skills = extract_skills(resume_text)
    #resume_vectorizer1 = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
    #resume_vectorized = resume_vectorizer1.fit_transform([resume_text])
    #resume_skills = set(resume_vectorizer1.get_feature_names_out())

    # Check if key skills in the job description are present in the resume
    key_skills_present = jd_skills.issubset(resume_skills)
    
    if jd_match >= 38 or key_skills_present:
        return "<b style='color: black;'>Yes</b>"
    else:
        return "<b style='color: black;'>No</b>"


st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
uploaded_job_description = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="job_description")
uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"], help="Please upload the resume in PDF format")

if uploaded_job_description is not None:
    st.write("Job Description PDF Uploaded Successfully")

if uploaded_resume is not None:
    st.write("Resume PDF Uploaded Successfully")

submit = st.button("Analyze Resume")

if submit:
    if uploaded_resume is not None and uploaded_job_description is not None:
        job_description_text = input_pdf_text(uploaded_job_description)
        resume_text = input_pdf_text(uploaded_resume)
        jd_match = calculate_jd_match(job_description_text, resume_text)
        position_match = assess_position_match(jd_match, resume_text, job_description_text)
        response = get_gemini_response(job_description_text, resume_text, jd_match, position_match)
        st.subheader("Analysis Result:")
        st.markdown(response, unsafe_allow_html=True)  # Use st.markdown to render HTML
    else:
        st.write("Please upload both the job description and the resume")
