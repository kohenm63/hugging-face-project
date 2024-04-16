# Import necessary libraries
import streamlit as st  # Streamlit library for creating web apps
from transformers import pipeline, logging  # Import pipeline for NLP tasks and logging for log management
import torch  # PyTorch, used by transformers for model computations

# Set logging level to error to minimize console output that isn't critical
logging.set_verbosity_error()

# Initialize the translation model using Hugging Face's pipeline
translator = pipeline(
    task="translation",  # Specify the task as 'translation'
    model="facebook/nllb-200-distilled-600M",  # Model identifier on Hugging Face
    torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency on compatible hardware
)

# Initialize the summarization model similarly using Hugging Face's pipeline
summarizer = pipeline(
    task="summarization",  # Task type is 'summarization'
    model="facebook/bart-large-cnn",  # Model identifier for a summarization model
    torch_dtype=torch.bfloat16  # Using bfloat16 to reduce memory usage
)

# Streamlit user interface setup for the web app
st.title('Multilingual Translation and Text Summarization')  # Set the title of the web app
st.subheader('Translation')  # Subheader for the translation section

# Text area for user to input text to translate with specified height
input_text = st.text_area("Enter text to translate:", height=150)

# Dropdown menu for selecting the source language
src_lang = st.selectbox("Select source language code (e.g., 'eng_Latn'):", ['eng_Latn', 'fra_Latn', 'spa_Latn'])

# Dropdown menu for selecting the target language
tgt_lang = st.selectbox("Select target language code (e.g., 'fra_Latn'):", ['fra_Latn', 'eng_Latn', 'spa_Latn'])

# Button to trigger translation; checks if text is entered before processing
if st.button('Translate'):
    if input_text:  # Ensure there is text entered before translating
        translated_text = translator(input_text, src_lang=src_lang, tgt_lang=tgt_lang)  # Perform translation
        # Display translated text in a text area with specified height
        st.text_area("Translated Text:", value=translated_text[0]['translation_text'], height=150)

# Subheader for the summarization section
st.subheader('Summarization')

# Text area for user to input text for summarization
input_text_summary = st.text_area("Enter text to summarize:", height=150)

# Button to trigger summarization; checks if text is entered before processing
if st.button('Summarize'):
    if input_text_summary:  # Ensure there is text to summarize
        summary_result = summarizer(input_text_summary, min_length=10, max_length=100)  # Perform summarization
        # Display summarized text in a text area
        st.text_area("Summary:", value=summary_result[0]['summary_text'], height=150)

# Garbage collection to clean up memory and prevent memory leaks
import gc
gc.collect()

