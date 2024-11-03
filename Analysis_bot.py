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
    openai.api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "openai_api_key" to your secrets.')
    st.stop()

def call_chatgpt(prompt, model="gpt-4", max_tokens=1000, temperature=0.3, retries=2):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
    Includes error handling and rate limiting.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]  # Ensures the JSON response is properly terminated
        )
        # Log the response for debugging (optional)
        logger.info(f"API Response: {response}")
        return response.choices[0].message.content.strip()
    except openai.RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            logger.error("Rate limit exceeded.")
            return ""
    except openai.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        return ""

def convert_to_markdown(data):
    """Converts the analysis data to Markdown format."""
    markdown = ""
    for theme in data.get("superordinate_themes", []):
        markdown += f"## {theme['superordinate_theme']}\n\n"
        for subtheme in theme.get("subthemes", []):
            markdown += f"### {subtheme['subtheme']}\n\n"
            markdown += f"**Description:** {subtheme.get('description', 'N/A')}\n\n"
            markdown += f"**Extracts:**\n"
            for extract in subtheme.get("extracts", []):
                markdown += f"- {extract}\n"
            markdown += "\n**Analytic Comments:**\n"
            for comment in subtheme.get("analytic_comments", []):
                markdown += f"- {comment}\n"
            markdown += "\n"
    return markdown

def convert_to_text(data):
    """Converts the analysis data to plain text format."""
    text = ""
    for theme in data.get("superordinate_themes", []):
        text += f"Superordinate Theme: {theme['superordinate_theme']}\n"
        for subtheme in theme.get("subthemes", []):
            text += f"  Subtheme: {subtheme['subtheme']}\n"
            text += f"    Description: {subtheme.get('description', 'N/A')}\n"
            text += f"    Extracts:\n"
            for extract in subtheme.get("extracts", []):
                text += f"      - {extract}\n"
            text += f"    Analytic Comments:\n"
            for comment in subtheme.get("analytic_comments", []):
                text += f"      - {comment}\n"
            text += "\n"
    return text

def save_output(data, file_path, format="json"):
    """Saves the data to a specified file format."""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if format == "json":
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        elif format == "markdown":
            with open(file_path, 'w', e
