from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import re
from sklearn.feature_extraction.text import CountVectorizer
import os
import spacy

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Define the prompt template
prompt_template = """
Job Description: {job_description}
Resume: {resume_text}
JD Match: {jd_match}
Education: {education}
Stability: {stability}

As an ATS scanner and a Technical HR Manager, please provide an analysis of the resume based on the job description with the following details:
- JD Match: {jd_match}%
- Experience: [years]
- Skills Missing: [skills missing keywords],
- Overall Summary: [brief summary]
- Position Match: {position_match}
- Education Match: {education}%
- Stability: {stability}%
"""

def get_gemini_response(job_description, resume_text, jd_match, position_match, education_match, stability_percentage):
    # Format the input text using the prompt template
    input_text = prompt_template.format(
        job_description=job_description,
        resume_text=resume_text,
        jd_match=jd_match,
        position_match=position_match,
        education=education_match,
        stability=stability_percentage
    )
    model = genai.GenerativeModel('gemini-pro')
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

    # Check if key skills in the job description are present in the resume
    key_skills_present = jd_skills.issubset(resume_skills)
    
    if jd_match >= 38 or key_skills_present:
        return "<b style='color: black;'>Yes</b>"
    else:
        return "<b style='color: black;'>No</b>"

#def extract_education(text):
    # Define a list of common education degrees or institutions
    #common_education = {'Bachelor', 'Master', 'PhD', 'B.Sc','HSC','University Degree','M.Sc','BSc', 'MSc', 'MBA', 'University', 'College'}
    # Use spaCy for NER to extract education entities
    #doc = nlp(text)
    #education_entities = {ent.text for ent in doc.ents if ent.label_ in ['ORG', 'NORP']}
    # Calculate the education match percentage
    #education_match = len(common_education.intersection(education_entities)) / len(common_education) * 100
    #return round(education_match, 2)

def extract_education(text):
    # Define a list of common education degrees, institutions, and keywords
    common_education_keywords = {
        'Bachelor', 'Master', 'BAF', 'S.S.C', 'PhD', 'SSC', 'S.Sc','B.Sc','H.S.C','B.Com', 'HSC','University Degree','M.Sc','BSc', 'MSc', 'MBA', 'University', 'College',
        'bachelor', 'master', 'phd', 'bsc', 'msc','hsc', 's.s.c', 'baf','h.s.c', 'b.com', 'mba', 'associate', 'diploma',
        'university', 'college', 'school', 'institute', 'faculty', 'department',
        'graduated', 'degree', 'honors', 'honours','Bachelor of Arts in Economics','Masters Of Arts in Economics'
    }

    # Convert the text to lowercase and use spaCy for NER to extract education entities
    text_lower = text.lower()
    doc = nlp(text_lower)

    # Extract entities and keywords related to education
    education_entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'NORP']}
    education_keywords = {word for word in text_lower.split() if word in common_education_keywords}

    # Combine the entities and keywords
    education_matches = education_entities.union(education_keywords)

    # Calculate the education match percentage (optional, depending on your requirements)
    if education_matches:
        education_match = len(education_matches) / len(common_education_keywords) * 100
    else:
        education_match = 0

    return round(education_match, 2)


def assess_stability(resume_text):
    # Use spaCy for NER to extract dates and calculate stability
    doc = nlp(resume_text)
    date_entities = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    # Placeholder stability calculation (you need to define your own logic here)
    stability = len(date_entities) * 10  # This is a placeholder
    # Convert stability to a percentage based on a predefined maximum value (e.g., 100% for 10 years)
    max_stability = 100  # Maximum stability value
    stability_percentage = min(stability, max_stability) / max_stability * 100
    return round(stability_percentage, 2)


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
        education_match = extract_education(resume_text)
        stability_percentage = assess_stability(resume_text)
        position_match = assess_position_match(jd_match, resume_text, job_description_text)
        response = get_gemini_response(job_description_text, resume_text, jd_match, position_match, education_match, stability_percentage)
        st.subheader("Analysis Result:")
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.write("Please upload both the job description and the resume")

