import streamlit as st
import openai
import openai.error  # Import the openai.error module
import json
import time
import os
import logging
from datetime import datetime

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

def generate_pets(transcript_text, participant_id):
    """Generates PETs for a single participant."""
    prompt = f"""
    You are to perform an Interpretative Phenomenological Analysis (IPA) on the following transcript for participant {participant_id}.
    Follow the process:

    1. Make initial notes or comments.
    2. Create experiential statements by refining notes into assertions.
    3. Develop Personal Experiential Themes (PETs) by clustering experiential statements.
    4. Review and refine the PETs.

    Provide the output in a well-formatted JSON structure as follows:
    {{
      "personal_experiential_themes": [
        {{
          "personal_experiential_theme": "Theme Title 1",
          "description": "Brief description of Theme Title 1",
          "extracts": [
            "Relevant extract from the transcript ({participant_id})",
            "Another relevant extract ({participant_id})"
          ],
          "analytic_comments": [
            "Analytic comment related to Theme Title 1",
            "Another analytic comment"
          ]
        }},
        ...
      ]
    }}

    Ensure the quotes are attributed to {participant_id} after each extract (e.g., "This experience was transformative." ({participant_id})).
    Do not include any text outside the JSON structure.

    Transcript:
    {transcript_text}
    """
    return call_chatgpt(prompt, max_tokens=2000)

def generate_gets(participant_pets):
    """Generates GETs by analyzing PETs across participants."""
    pets_across_participants = []
    for participant in participant_pets:
        participant_id = participant['participant_id']
        pets = participant['pets'].get('personal_experiential_themes', [])
        for pet in pets:
            pet['participant_id'] = participant_id
            pets_across_participants.append(pet)

    prompt = f"""
    You are to perform an Interpretative Phenomenological Analysis (IPA) to develop Group Experiential Themes (GETs) by analyzing the following Personal Experiential Themes (PETs) across participants.
    Follow the process:

    1. Review the PETs from all participants.
    2. Identify patterns of convergence and divergence.
    3. Develop Group Experiential Themes (GETs) that represent the group, including subthemes if appropriate.
    4. Ensure that GETs highlight shared and unique features of the experience across participants.

    Provide the output in a well-formatted JSON structure as follows:
    {{
      "group_experiential_themes": [
        {{
          "group_experiential_theme": "GET Title 1",
          "description": "Brief description of GET Title 1",
          "subthemes": [
            {{
              "subtheme": "Subtheme Title",
              "description": "Description of Subtheme",
              "participant_contributions": [
                {{
                  "participant_id": "P1",
                  "extracts": ["Extract from participant"],
                  "analytic_comments": ["Comment from participant"]
                }},
                ...
              ]
            }},
            ...
          ]
        }},
        ...
      ]
    }}

    Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.

    Personal Experiential Themes (PETs) from participants:
    {json.dumps(pets_across_participants, indent=2)}
    """
    return call_chatgpt(prompt, max_tokens=2000)

def stage4_write_up_pet(participant_pets, transcripts):
    """Stage 4: Writing up PETs with extracts and analytic comments."""
    prompt = f"""
    Using the following Personal Experiential Themes (PETs) from an IPA analysis, write up each theme concisely.
    For each PET, include:
    - A brief description.
    - Relevant extracts from the transcripts (limit to 2 extracts per participant).
    - Analytic comments (maximum of 2 per theme).

    Ensure that quotes are attributed to participants (e.g., "This experience was transformative." (P1)).

    Provide the output in a well-formatted JSON structure as follows:
    {{
        "personal_experiential_themes": [
            {{
                "personal_experiential_theme": "Theme Title 1",
                "description": "Brief description of Theme Title 1",
                "extracts": [
                    "Relevant extract from the transcript (P1)",
                    "Another relevant extract (P2)"
                ],
                "analytic_comments": [
                    "Analytic comment related to Theme Title 1",
                    "Another analytic comment"
                ]
            }},
            ...
        ]
    }}

    Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.

    Personal Experiential Themes (PETs):
    {json.dumps(participant_pets, indent=2)}

    Transcripts:
    {transcripts}
    """
    return call_chatgpt(prompt, max_tokens=2000)

def convert_analysis_to_markdown(analysis):
    """Converts the analysis data to Markdown format, including PETs and GETs."""
    markdown = ""

    # Process Stage 4 PETs Write-up
    pets_writeup = analysis.get('personal_experiential_themes_writeup', [])
    if pets_writeup:
        markdown += "# Personal Experiential Themes (Write-up)\n\n"
        for pet in pets_writeup:
            pet_title = pet.get('personal_experiential_theme', 'Untitled Theme')
            markdown += f"## {pet_title}\n\n"
            description = pet.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            extracts = pet.get("extracts", [])
            markdown += f"**Extracts:**\n"
            if extracts:
                for extract in extracts:
                    markdown += f"- {extract}\n"
            else:
                markdown += "- No extracts provided.\n"
            analytic_comments = pet.get("analytic_comments", [])
            markdown += "\n**Analytic Comments:**\n"
            if analytic_comments:
                for comment in analytic_comments:
                    markdown += f"- {comment}\n"
            else:
            
