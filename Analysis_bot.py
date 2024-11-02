import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenAI class
from openai import OpenAI

# Initialize OpenAI Client with the API key
client = OpenAI(
    api_key=st.secrets["openai_api_key"]  # Ensure this key exists in Streamlit secrets
)

def call_chatgpt(prompt, model="gpt-4", max_tokens=800, temperature=0.3, retries=3):
    """
    Sends a prompt to the OpenAI ChatGPT API using the client-based interface and returns the response.
    Includes basic error handling and rate limiting.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA).",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Log the response for debugging (optional)
        logger.info(f"API Response: {response}")
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            logger.error("Rate limit exceeded.")
            return ""
    except openai.error.OpenAIError as e:
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
    for pet in data.get("personal_experiential_themes", []):
        markdown += f"## {pet['personal_experiential_theme']}\n\n"
        markdown += f"**Description:** {pet.get('description', 'N/A')}\n\n"
        markdown += f"**Extracts:**\n"
        for extract in pet.get("extracts", []):
            markdown += f"- {extract}\n"
        markdown += "\n**Analytic Comments:**\n"
        for comment in pet.get("analytic_comments", []):
            markdown += f"- {comment}\n"
        markdown += "\n"
    return markdown

def convert_to_text(data):
    """Converts the analysis data to plain text format."""
    text = ""
    for pet in data.get("personal_experiential_themes", []):
        text += f"Personal Experiential Theme (PET): {pet['personal_experiential_theme']}\n"
        text += f"  Description: {pet.get('description', 'N/A')}\n"
        text += f"  Extracts:\n"
        for extract in pet.get("extracts", []):
            text += f"    - {extract}\n"
        text += f"  Analytic Comments:\n"
        for comment in pet.get("analytic_comments", []):
            text += f"    - {comment}\n"
        text += "\n"
    return text

def save_output(data, file_path, format="markdown"):
    """Saves the data to a specified file format."""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if format == "json":
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        elif format == "markdown":
            with open(file_path, "w", encoding="utf-8") as file:
                markdown_content = convert_to_markdown(data)
                file.write(markdown_content)
        elif format == "text":
            with open(file_path, "w
