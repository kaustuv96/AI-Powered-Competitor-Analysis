#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries

# In[ ]:


import os
import streamlit as st
import langchain
import transformers
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
import re
from transformers import AutoTokenizer
import PyPDF2
from io import BytesIO


# #### Printing library versions

# In[ ]:


# ### Printing the Versions of Libraries Used

# In[3]:


# Print library versions
print("Libraries used:")
print(f"- streamlit: {st.__version__}")
print(f"- langchain: {langchain.__version__}")
print(f"- transformers: {transformers.__version__}")
print(f"- PyPDF2: {PyPDF2.__version__}")
print("- re") # Built-in module, no version
print("- os") # Built-in module, no version
print("- io") #Built-in module, no version


# In[ ]:


# Google Generative AI (langchain_google_genai) version: Version information not available


# #### Initial steps

# In[ ]:


# ### Initial Steps

# In[ ]:


# --- Streamlit App Configuration ---
st.set_page_config(page_title="AI-Powered Competitor Analyzer")

# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Retrieves the Google API key from the environment variables.
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set the GOOGLE_API_KEY environment variable.")
    st.stop()

# --- Initialize Gemini Model ---
gemini_llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY) # Initializes the Gemini LLM with specified parameters.

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file): # Defines a function to extract text from PDF files.
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process long text by splitting and summarizing
def process_long_text(text): # Defines a function to process long texts by splitting and summarizing.
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # For multiple chunks, summarize first
    if len(chunks) > 1:
        model = get_model()
        summaries = []
        for chunk in chunks:
            response = model.generate_content(f"Summarize the following text concisely:\n\n{chunk}")
            summaries.append(response.text)
        
        return " ".join(summaries)
    else:
        return text


# #### Creating prompt template

# In[ ]:


# --- Prompt Templates ---
competitor_analysis_template = """
You are an expert market analyst. Provide a competitor analysis based on the provided text. Focus on identifying the key competitors mentioned, 
their strengths, weaknesses, market strategies, potential opportunities, and areas of differentiation as described in the text. Format each section with clear headings.
Also include possible innovation or disruption threats that the competitors face, based on the text. Provide specific examples where possible.

Text:
{text}

Competitor Analysis:
"""

competitor_prompt = PromptTemplate(input_variables=["text"], template=competitor_analysis_template)
competitor_chain = LLMChain(prompt=competitor_prompt, llm=gemini_llm, output_key="competitor_analysis")


# #### Creating Token Tracker

# In[ ]:


# --- Token Tracking Variables ---
input_tokens_used = 0
output_tokens_used = 0

# Initialize the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # or a suitable Gemini tokenizer
except Exception as e:
    st.warning(f"Failed to load tokenizer: {e}. Token counts will be a rough estimate.")
    tokenizer = None


def count_tokens(text: str, tokenizer) -> int:
    """Counts the number of tokens in a text string using a Hugging Face tokenizer."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            st.warning(f"Failed to count tokens using tokenizer: {e}. Using a rough estimate (word count).")
            return len(text.split())  # Fallback to a rough estimate if tokenization fails
    else:
        return len(text.split())  # Fallback to a rough estimate if no tokenizer is available

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def update_token_display(input_used=None, output_used=None):
    """Updates the token usage display in the sidebar."""
    markdown_text = ""
    if input_used is not None:
        markdown_text += f"""**Input Tokens Used:** {input_used:,}  \n"""
    if output_used is not None:
        markdown_text += f"""**Output Tokens Used:** {output_used:,}"""

    st.sidebar.markdown(markdown_text, unsafe_allow_html=True)


# #### Setting Up the UI

# In[ ]:


# --- Streamlit App Layout ---
st.title("AI-Powered Competitor Analyzer")
st.markdown("An AI-powered Agent using Streamlit and LangChain to analyze business information, do competitor analysis.")

st.markdown("Powered by Gemini-2.0-Flash, LangChain version: 0.3.20, & Streamlit version: 1.37.1")
st.markdown("Transformers version: 4.49.0, Google Generative AI version: Version information not available")

st.markdown("Upload a file with competitor data and relevant business information or enter text to generate a competitor analysis.")

uploaded_file = st.file_uploader("Choose a text file", type=["txt", "pdf", "csv"])
text_input = st.text_area("Or enter text directly:", height=200)


text = ""  # Initialize text

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
    else:
        try:
            text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Error: The uploaded text file is not UTF-8 encoded. Please ensure the file is properly encoded.")
            text = None
else:
    text = text_input if text_input else None

if text:
    st.subheader("Original Text:")
    st.markdown(f'<div style="height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">{text}</div>', unsafe_allow_html=True)

if st.button("Analyze Competitors"):
    if text:
        with st.spinner("Analyzing..."):
            try:
                # Track Input Tokens. Text used as Input
                input_text = f"Text: {text}" #All inputs measured here
                input_tokens_used = count_tokens(input_text, tokenizer)
                update_token_display(input_used=input_tokens_used)

                analysis = competitor_chain.run(text=text) #Pass text to chain

                # Track Output Tokens
                output_tokens_used = count_tokens(analysis, tokenizer)
                update_token_display(output_used=output_tokens_used)

                st.subheader("Competitor Analysis Report:")
                st.write(analysis)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please provide text input.")

