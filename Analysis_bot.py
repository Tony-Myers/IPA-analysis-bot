import streamlit as st
import openai
import json
import time
import logging
import re

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API Key
try:
    api_key = st.secrets["openai_api_key"]
    openai.api_key = api_key
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "openai_api_key" to your secrets.')
    st.stop()

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.0, retries=2):
    """
    Calls the OpenAI API and parses the JSON response.
    """
    try:
        # Ensure proper API usage with ChatCompletion
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]
        )
        # Process the response as needed
        message_content = response.choices[0].message.get("content", "{}")
        return json.loads(fix_json(message_content))

    except Exception as e:
        # Handle rate limits and other errors in a general way
        if "Rate limit" in str(e) and retries > 0:
            st.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error(f"API error: {e}")
            return {}

