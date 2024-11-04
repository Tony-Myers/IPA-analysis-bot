import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    api_key = st.secrets["openai_api_key"]
    client = openai.OpenAI(api_key=api_key)
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "openai_api_key" to your secrets.')
    st.stop()

def call_chatgpt(prompt, model="gpt-4", max_tokens=1000, temperature=0.3, retries=2):
    """
    Calls the OpenAI API and handles JSON parsing errors.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in IPA."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]
        )
        logger.info(f"API Response: {response}")
        # Attempt JSON loading here to ensure valid JSON is returned
        try:
            content = response.choices[0].message.content.strip()
            # Ensure content is valid JSON before returning
            json.loads(content)  # Test JSON formatting
            return content
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return ""  # Empty string indicates a parsing issue
    except openai.error.RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded.")
            return ""
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""

# Modified stages to handle JSON errors explicitly and gracefully
def stage1_initial_notes(transcript):
    prompt = f"..."
    result = call_chatgpt(prompt)
    try:
        return json.loads(result) if result else {}
    except json.JSONDecodeError:
        st.error("Stage 1 JSON parsing failed. Check the output format.")
        logger.error("Stage 1 JSON parsing failed.")
        return {}

def stage2_experiential_statements(initial_notes):
    prompt = f"..."
    result = call_chatgpt(prompt)
    try:
        return json.loads(result) if result else []
    except json.JSONDecodeError:
        st.error("Stage 2 JSON parsing failed. Check the output format.")
        logger.error("Stage 2 JSON parsing failed.")
        return []

def stage3_cluster_pet(es):
    prompt = f"..."
    result = call_chatgpt(prompt)
    try:
        return json.loads(result) if result else {}
    except json.JSONDecodeError:
        st.error("Stage 3 JSON parsing failed. Check the output format.")
        logger.error("Stage 3 JSON parsing failed.")
        return {}

def stage4_get_writeup(pets, transcript):
    prompt = f"..."
    result = call_chatgpt(prompt)
    try:
        return json.loads(result) if result else {}
    except json.JSONDecodeError:
        st.error("Stage 4 JSON parsing failed. Check the output format.")
        logger.error("Stage 4 JSON parsing failed.")
        return {}

# Updated `ipa_analysis_pipeline` to handle parsing issues better
def ipa_analysis_pipeline(transcript, output_path):
    try:
        transcript_text = transcript.read().decode("utf-8").strip()
        if not transcript_text:
            st.error("The uploaded transcript is empty.")
            return
    except Exception as e:
        st.error(f"Error reading the transcript file: {e}")
        logger.error(f"Error reading the transcript file: {e}")
        return
    
    st.write("### Stage 1: Generating Initial Notes...")
    with st.spinner("Generating initial notes..."):
        initial_notes = stage1_initial_notes(transcript_text)
    
    if not initial_notes:
        st.error("Stage 1 failed. Analysis incomplete.")
        return

    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    with st.spinner("Extracting ES..."):
        es = stage2_experiential_statements(initial_notes)
    
    if not es:
        st.error("Stage 2 failed. Analysis incomplete.")
        return

    st.write("### Stage 3: Clustering PETs...")
    with st.spinner("Clustering PETs..."):
        pets = stage3_cluster_pet(es)
    
    if not pets:
        st.error("Stage 3 failed. Analysis incomplete.")
        return

    st.write("### Stage 4: Writing up GETs...")
    with st.spinner("Writing up GETs..."):
        get_writeup = stage4_get_writeup(pets, transcript_text)
    
    if get_writeup:
        st.write("### Saving Final Analysis to Markdown...")
        save_output(get_writeup, output_path, format="markdown")
        st.markdown(convert_to_markdown(get_writeup))
    else:
        st.error("Stage 4 failed. Analysis incomplete.")

# Run the app
if __name__ == "__main__":
    main()
