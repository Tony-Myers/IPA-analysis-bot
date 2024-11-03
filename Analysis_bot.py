import streamlit as st
import openai
import json
import time
import os
import logging
from openai.error import OpenAIError, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the OpenAI API key
openai.api_key = st.secrets["openai_api_key"]  # Ensure this key exists in Streamlit secrets

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.3, retries=3):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
    Includes basic error handling and rate limiting.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA).",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Log the response for debugging
        logger.info(f"API Response: {response}")
        return response.choices[0].message.content.strip()
    except RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            logger.error("Rate limit exceeded.")
            return ""
    except OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        return ""

# The rest of your code remains the same
# Ensure that any references to 'client' are updated to use 'openai' directly

def generate_pets(transcript, participant_id):
    """Generates PETs for a single participant."""
    prompt = f"""
    [Your detailed prompt for generating PETs based on the instructions you provided]
    """
    return call_chatgpt(prompt, max_tokens=3000)

def generate_gets(pets_dict):
    """Generates GETs from PETs of all participants."""
    pets_json = json.dumps(pets_dict, indent=2)
    prompt = f"""
    [Your detailed prompt for generating GETs based on the instructions you provided]
    """
    return call_chatgpt(prompt, max_tokens=3000)

# ... rest of your functions ...

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write(
        """
    Upload your interview transcripts to perform IPA using ChatGPT.
    """
    )

    uploaded_files = st.file_uploader("Choose transcript text files", type=["txt"], accept_multiple_files=True)

    if st.button("Run IPA Analysis"):
        if uploaded_files:
            ipa_analysis_pipeline(uploaded_files)
        else:
            st.warning("Please upload transcript files.")

if __name__ == "__main__":
    main()
