import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client with the API key from Streamlit secrets
try:
    openai_api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("API key not found. Please add your OpenAI API key to Streamlit secrets.")
    st.stop()  # Stop the app if the API key isn't set up

def call_chatgpt(prompt, model="gpt-4", max_tokens=150, temperature=0.3, retries=2):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info(f"API Response: {response}")
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""

def stage4_write_up_themes(pets, transcript, retries=2):
    """Stage 4: Writing up themes with extracts and analytic comments."""
    prompt = f"""
    Using the following clustered themes, write up each theme with description, extracts, and comments.
    PETs: {json.dumps(pets, indent=2)}
    Transcript: {transcript}
    Output JSON: group_experiential_theme -> personal_experiential_theme -> description, extracts, analytic_comments.
    """
    
    for attempt in range(retries):
        response_text = call_chatgpt(prompt)
        try:
            # Attempt to parse JSON response
            response_json = json.loads(response_text)
            return response_json  # Return parsed JSON if successful
        except json.JSONDecodeError:
            # Log and retry parsing with modified max_tokens on the last retry
            if attempt < retries - 1:
                st.warning("Retrying Stage 4 with adjusted parameters...")
                logger.warning("Retrying Stage 4 with adjusted parameters due to JSON parsing error.")
                prompt = prompt[:2000]  # Shorten prompt to prevent length issues on retry
            else:
                st.error("Failed to parse JSON from Stage 4 after retries.")
                logger.error("JSON parsing failed for Stage 4 even after retries.")
                return {}

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")
    
    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])
    output_path = st.text_input("Enter the output path without extension (e.g., results/output_analysis)")

    if st.button("Run IPA Analysis"):
        if uploaded_file and output_path:
            transcript_text = uploaded_file.read().decode("utf-8")
            output_path = os.path.splitext(output_path)[0]
            # Run analysis pipeline and save results
            st.write("### Stage 4: Writing Up Themes with Extracts and Comments...")
            results = stage4_write_up_themes(transcript_text, output_path)
            if results:
                st.json(results)  # Display the final structured output for verification
        else:
            st.warning("Please upload a transcript file and specify an output path.")

if __name__ == "__main__":
    main()
