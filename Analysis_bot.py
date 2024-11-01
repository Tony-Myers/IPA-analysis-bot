import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.3):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
    Includes basic error handling and rate limiting.
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
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
        logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        return call_chatgpt(prompt, model, max_tokens, temperature)
    except openai.error.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        return ""

def stage1_initial_notes(transcript):
    """Stage 1: Close reading and initial notes."""
    prompt = f"""
    I want you to perform Stage 1 of Interpretative Phenomenological Analysis (IPA) on the following interview transcript. 
    This involves close reading of the transcript multiple times, making notes about observations, reflections, content, language use, context, and initial interpretative comments. 
    Highlight distinctive phrases and emotional responses. Include any personal reflexivity comments if relevant.

    Transcript:
    {transcript}

    Please provide your output in a structured JSON format with the following fields:
    - observations
    - reflections
    - content_notes
    - language_use
    - context
    - interpretative_comments
    - distinctive_phrases
    - emotional_responses
    - reflexivity_comments
    """
    return call_chatgpt(prompt)

def stage2_emergent_themes(initial_notes):
    """Stage 2: Transforming notes into emergent themes."""
    prompt = f"""
    Using the following initial notes from an IPA analysis, transform them into emergent themes. 
    Formulate concise phrases at a higher level of abstraction grounded in the participant’s account.

    Initial Notes:
    {json.dumps(initial_notes, indent=2)}

    Please provide the emergent themes in a JSON array format.
    """
    return call_chatgpt(prompt)

def stage3_cluster_themes(emergent_themes):
    """Stage 3: Seeking relationships and clustering themes."""
    prompt = f"""
    Based on the following emergent themes from an IPA analysis, identify connections between them, group them into clusters based on conceptual similarities, and organize them into superordinate themes and subthemes.

    Emergent Themes:
    {json.dumps(emergent_themes, indent=2)}

    Please provide the clustered themes in a structured JSON format with the following hierarchy:
    - superordinate_theme
        - subtheme
    """
    return call_chatgpt(prompt)

def stage4_write_up_themes(clustered_themes, transcript):
    """Stage 4: Writing up themes with extracts and analytic comments."""
    prompt = f"""
    Using the following clustered themes from an IPA analys
