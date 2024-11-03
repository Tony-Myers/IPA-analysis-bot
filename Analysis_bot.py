import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client with the API key
client = OpenAI(
    api_key=st.secrets["openai_api_key"]  # Ensure this key exists in Streamlit secrets
)

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.3, retries=3):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
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

def generate_pets(participant_id, transcript_text):
    """Generates Personal Experiential Themes (PETs) for a participant."""
    prompt = f"""
As an expert in Interpretative Phenomenological Analysis (IPA), please perform an analysis on the following interview transcript for participant {participant_id}.

Process:
- Make initial notes or comments.
- Create experiential statements – refine notes into assertions.
- Develop Personal Experiential Themes (PETs) – trends that reflect the commonalities – conceptual, theoretical, semantic, practical, etc. – by clustering experiential statements.
- Review and refine the PETs.

When naming PETs, give clusters of experiential statements a title that describes their characteristics. PET names or labels should be data and cluster driven.

Provide the output in a well-formatted JSON structure as follows:
{{
    "participant_id": "{participant_id}",
    "pets": [
        {{
            "pet_title": "Title of PET 1",
            "description": "Brief description of PET 1",
            "quotes": [
                "Short relevant quote from participant"
            ]
        }},
        ...
    ]
}}

Ensure that the quotes are attributed to the participant (e.g., '{participant_id}' in brackets after the quote).

Transcript:
{transcript_text}

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.
"""
    return call_chatgpt(prompt, max_tokens=3000)

def generate_gets(all_pets):
    """Generates Group Experiential Themes (GETs) by analyzing PETs across participants."""
    prompt = f"""
As an expert in Interpretative Phenomenological Analysis (IPA), please perform a cross-case analysis on the following Personal Experiential Themes (PETs) to develop Group Experiential Themes (GETs).

Process:
- Working with PETs from each participant, create Group Experiential Themes (GETs) by looking across individual cases for patterns of convergence and divergence.
- GETs should highlight the shared and unique features of the experience across the participants.
- When naming GETs, choose a label that captures each GET overall. Avoid using quotes from individual participants in the GET titles to prevent imposing one participant's experience onto others.

Provide the output in a well-formatted JSON structure as follows:
{{
    "gets": [
        {{
            "get_title": "Title of GET 1",
            "description": "Brief description of GET 1",
            "subthemes": [
                {{
                    "subtheme_title": "Title of Subtheme",
                    "participants": ["P1", "P2", ...],
                    "quotes": [
                        {{
                            "participant_id": "P1",
                            "quote": "Relevant quote from participant"
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

Use quotes to illustrate each GET and subtheme, and attribute quotes to participants (e.g., 'P1').

Personal Experiential Themes (PETs):
{json.dumps(all_pets, indent=2)}

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.
"""
    return call_chatgpt(prompt, max_tokens=4000)

def generate_final_report(all_pets, gets):
    """Generates the final report including PETs and GETs in Markdown format."""
    markdown = "# Interpretative Phenomenological Analysis Report\n\n"

    # Include PETs per participant
    for participant_id, pet_data in all_pets.items():
        markdown += f"## Personal Experiential Themes for {participant_id}\n\n"
        pets = pet_data.get('pets', [])
        if not pets:
            markdown += "No Personal Experiential Themes found.\n\n"
            continue
        for pet in pets:
            pet_title = pet.get('pet_title', 'Untitled Theme')
            markdown += f"### {pet_title}\n\n"
            description = pet.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            quotes = pet.get('quotes', [])
            if quotes:
                markdown += "**Quotes:**\n"
                for quote in quotes:
                    markdown += f"> {quote} ({participant_id})\n\n"
            else:
                markdown += "No quotes provided.\n\n"

    # Include GETs
    markdown += "## Group Experiential Themes\n\n"
    gets_list = gets.get('gets', [])
    if not gets_list:
        markdown += "No Group Experiential Themes found.\n\n"
    else:
        for get in gets_list:
            get_title = get.get('get_title', 'Untitled GET')
            markdown += f"### {get_title}\n\n"
            description = get.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            subthemes = get.get('subthemes', [])
            for subtheme in subthemes:
                subtheme_title = subtheme.get('subtheme_title', 'Untitled Subtheme')
                participants = subtheme.get('participants', [])
                participants_str = ', '.join(participants)
                markdown += f"#### {subtheme_title} (Participants: {participants_str})\n\n"
                quotes = subtheme.get('quotes', [])
                if quotes:
                    markdown += "**Quotes:**\n"
                    for quote_data in quotes:
                        participant_id = quote_data.get('participant_id', '')
                        quote = quote_data.get('quote', '')
                        markdown += f"> {quote} ({participant_id})\n\n"
                else:
                    markdown += "No quotes provided.\n\n"
    return markdown

def ipa_analysis_pipeline(uploaded_files):
    """Runs the full IPA analysis pipeline on the given transcripts."""
    transcripts = {}
    for idx, file in enumerate(uploaded_files):
        participant_id = f"P{idx+1}"
        try:
            transcript_text = file.read().decode("utf-8")
            if not transcript_text.strip():
                st.error(f"The uploaded transcript file {file.name} is empty.")
                return
            transcripts[participant_id] = transcript_text
        except Exception as e:
            st.error(f"Error reading the transcript file {file.name}: {e}")
            logger.error(f"Error reading the transcript file {file.name}: {e}")
            return

    # Now, process each transcript individually to generate PETs
    st.write("### Generating Personal Experiential Themes (PETs) for each participant...")
    all_pets = {}
    for participant_id, transcript_text in transcripts.items():
        st.write(f"Processing {participant_id}...")
        with st.spinner(f"Generating PETs for {participant_id}..."):
            pet_json = generate_pets(participant_id, transcript_text)
            if pet_json:
                try:
                    pets = json.loads(pet_json)
                    all_pets[participant_id] = pets
                    st.success(f"Generated PETs for {participant_id}.")
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON for {participant_id}.")
                    logger.error(f"Error parsing JSON for {participant_id}: {e}")
                    st.write("API Response Content:")
                    st.code(pet_json)
            else:
                st.error(f"Failed to generate PETs for {participant_id}.")

    # After generating PETs for all participants, generate GETs
    st.write("### Generating Group Experiential Themes (GETs)...")
    with st.spinner("Generating GETs..."):
        get_json = generate_gets(all_pets)
        if get_json:
            try:
                gets = json.loads(get_json)
                st.success("Generated GETs.")
            except json.JSONDecodeError as e:
                st.error("Error parsing JSON for GETs.")
                logger.error(f"Error parsing JSON for GETs: {e}")
                st.write("API Response Content:")
                st.code(get_json)
        else:
            st.error("Failed to generate GETs.")

    # Now, compile the final report including both PETs and GETs
    st.write("### Final Analysis:")
    markdown_content = generate_final_report(all_pets, gets)
    if not markdown_content:
        st.error("Failed to generate Markdown content.")
        return

    # Keep the results visible until the user decides to close them
    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = True

    if st.session_state['show_results']:
        st.markdown(markdown_content)
        if st.button("Hide Results"):
            st.session_state['show_results'] = False
    else:
        if st.button("Show Results"):
            st.session_state['show_results'] = True

    # Provide download button for markdown file only
    st.write("### Download Results:")
    markdown_bytes = markdown_content.encode("utf-8")
    st.download_button(
        label="Download Analysis as Markdown",
        data=markdown_bytes,
        file_name="ipa_analysis.md",
        mime="text/markdown",
    )

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write(
        """
        Upload your interview transcripts to perform IPA using ChatGPT.
        """
    )

    uploaded_files = st.file_uploader(
        "Choose transcript text files",
        type=["txt"],
        accept_multiple_files=True,
    )

    if st.button("Run IPA Analysis"):
        if uploaded_files:
            ipa_analysis_pipeline(uploaded_files)
        else:
            st.warning("Please upload at least one transcript file.")

if __name__ == "__main__":
    main()
