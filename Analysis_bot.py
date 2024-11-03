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

def generate_pets(transcript, participant_id):
    """
    Processes a single transcript to generate PETs.
    """
    prompt = f"""
You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA).

Analyze the following interview transcript from participant {participant_id} to generate Personal Experiential Themes (PETs).

Follow the IPA analysis process:
- Make initial notes or comments
- Create experiential statements
- Develop personal experiential themes (PETs) by clustering experiential statements
- Review and refine the PETs

For each PET, provide:
- PET title (name the PET appropriately, reflecting its characteristics)
- Brief description
- Short relevant sections of participant quotes to illustrate the PET (maximum of 2 quotes per PET)
- Attribute each quote to the participant (e.g., {participant_id})

Provide the output in a well-formatted JSON structure as follows:
{{
  "participant_id": "{participant_id}",
  "pets": [
    {{
      "personal_experiential_theme": "PET Title",
      "description": "Brief description of the PET",
      "quotes": [
        "Quote 1",
        "Quote 2"
      ]
    }},
    ...
  ]
}}

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.

Transcript:
{transcript}
"""
    # Call the OpenAI API
    response = call_chatgpt(prompt, max_tokens=1500)
    return response

def generate_gets(all_pets):
    """
    Processes all PETs to generate GETs.
    """
    pets_json = json.dumps(all_pets, indent=2)

    prompt = f"""
You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA).

Using the Personal Experiential Themes (PETs) from multiple participants, identify Group Experiential Themes (GETs).

Follow the IPA analysis process:
- Working with PETs, create group experiential themes (GETs) by looking across individual cases for patterns of convergence and divergence.

For each GET, provide:
- GET title (name the GET appropriately, reflecting shared and unique features of the experience across participants)
- Brief description
- Sub-themes within the GET (if applicable)
- Short relevant sections of participant quotes to illustrate the GET (maximum of 2 quotes per GET)
- Attribute each quote to the participant (e.g., P1, P2, etc.)

Provide the output in a well-formatted JSON structure as follows:
{{
  "group_experiential_themes": [
    {{
      "group_experiential_theme": "GET Title",
      "description": "Brief description of the GET",
      "subthemes": [
        {{
          "subtheme_title": "Subtheme Title",
          "description": "Brief description of the subtheme",
          "quotes": [
            "Quote 1",
            "Quote 2"
          ],
          "participants": ["P1", "P2"]
        }},
        ...
      ],
      "quotes": [
        "Quote 1",
        "Quote 2"
      ],
      "participants": ["P1", "P2", "P3"]
    }},
    ...
  ]
}}

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.

Personal Experiential Themes (PETs) from participants:
{pets_json}
"""
    # Call the OpenAI API
    response = call_chatgpt(prompt, max_tokens=1500)
    return response

def convert_to_markdown_pets_and_gets(all_pets, get_data):
    """Converts the PETs and GETs data to Markdown format."""
    markdown = "# Interpretative Phenomenological Analysis (IPA) Report\n\n"

    # Include PETs for each participant
    for pet_data in all_pets:
        participant_id = pet_data.get("participant_id", "Unknown Participant")
        pets = pet_data.get("pets", [])
        if not pets:
            st.error(f"No PETs found for {participant_id}.")
            logger.error(f"No PETs found for {participant_id}.")
            continue
        markdown += f"## Personal Experiential Themes (PETs) for {participant_id}\n\n"
        for pet in pets:
            pet_title = pet.get('personal_experiential_theme', 'Untitled Theme')
            markdown += f"### {pet_title}\n\n"
            description = pet.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            quotes = pet.get('quotes', [])
            markdown += f"**Quotes from {participant_id}:**\n"
            if quotes:
                for quote in quotes:
                    markdown += f"> \"{quote}\"  \n"
            else:
                markdown += "> No quotes provided.\n"
            markdown += "\n"

    # Include GETs
    group_experiential_themes = get_data.get("group_experiential_themes", [])
    if not group_experiential_themes:
        st.error("No Group Experiential Themes (GETs) found.")
        logger.error("No GETs found.")
    else:
        markdown += "## Group Experiential Themes (GETs)\n\n"
        for get in group_experiential_themes:
            get_title = get.get('group_experiential_theme', 'Untitled GET')
            markdown += f"### {get_title}\n\n"
            description = get.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            quotes = get.get('quotes', [])
            markdown += f"**Quotes from Participants:**\n"
            if quotes:
                for quote in quotes:
                    markdown += f"> \"{quote}\"  \n"
            else:
                markdown += "> No quotes provided.\n"
            markdown += "\n"
            # Include subthemes if any
            subthemes = get.get('subthemes', [])
            if subthemes:
                for subtheme in subthemes:
                    subtheme_title = subtheme.get('subtheme_title', 'Untitled Subtheme')
                    markdown += f"#### Subtheme: {subtheme_title}\n\n"
                    sub_description = subtheme.get('description', 'N/A')
                    markdown += f"**Description:** {sub_description}\n\n"
                    sub_quotes = subtheme.get('quotes', [])
                    participants = subtheme.get('participants', [])
                    participant_str = ', '.join(participants)
                    markdown += f"**Quotes from Participants ({participant_str}):**\n"
                    if sub_quotes:
                        for quote in sub_quotes:
                            markdown += f"> \"{quote}\"  \n"
                    else:
                        markdown += "> No quotes provided.\n"
                    markdown += "\n"

    return markdown

def ipa_analysis_pipeline(transcripts, output_filename):
    """Runs the full IPA analysis pipeline on given transcripts."""

    transcripts_text = {}
    participant_ids = []
    all_pets = []
    for idx, transcript_file in enumerate(transcripts):
        participant_id = f"P{idx+1}"
        participant_ids.append(participant_id)
        try:
            transcript_content = transcript_file.read().decode("utf-8")
            if not transcript_content.strip():
                st.error(f"The uploaded transcript file {transcript_file.name} is empty.")
                return
            transcripts_text[participant_id] = transcript_content
        except Exception as e:
            st.error(f"Error reading the transcript file {transcript_file.name}: {e}")
            logger.error(f"Error reading the transcript file {transcript_file.name}: {e}")
            return
        # Display the transcript in an expander
        with st.expander(f"Transcript for {participant_id} ({transcript_file.name})"):
            st.text(transcript_content)

    # For each participant, generate PETs
    st.write("### Generating Personal Experiential Themes (PETs) for each participant...")
    for participant_id in participant_ids:
        st.write(f"Processing {participant_id}...")
        with st.spinner(f"Generating PETs for {participant_id}..."):
            pet_response = generate_pets(transcripts_text[participant_id], participant_id)
            if pet_response:
                try:
                    pet_data = json.loads(pet_response)
                    st.success(f"Generated PETs for {participant_id}.")
                    all_pets.append(pet_data)
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON for {participant_id}. Please check the API response.")
                    logger.error(f"Error parsing JSON for {participant_id}: {e}")
                    st.write("API Response Content:")
                    st.code(pet_response)
            else:
                st.error(f"Failed to generate PETs for {participant_id}.")
                return

    # Generate GETs based on all PETs
    st.write("### Generating Group Experiential Themes (GETs)...")
    with st.spinner("Generating GETs..."):
        get_response = generate_gets(all_pets)
        if get_response:
            try:
                get_data = json.loads(get_response)
                st.success("Generated GETs.")
            except json.JSONDecodeError as e:
                st.error("Error parsing JSON for GETs. Please check the API response.")
                logger.error(f"Error parsing JSON for GETs: {e}")
                st.write("API Response Content:")
                st.code(get_response)
                get_data = {}
        else:
            st.error("Failed to generate GETs.")
            return

    # Prepare final analysis report including PETs and GETs
    st.write("### Final Analysis:")
    markdown_content = convert_to_markdown_pets_and_gets(all_pets, get_data)
    if not markdown_content:
        st.error("Failed to generate Markdown content.")
        return
    st.markdown(markdown_content)

    # Provide download button for markdown
    st.write("### Download Results:")
    # Prepare file name
    markdown_filename = f"{output_filename}.md"
    # Encode the content
    markdown_bytes = markdown_content.encode("utf-8")
    # Create download button
    st.download_button(
        label="Download Markdown",
        data=markdown_bytes,
        file_name=markdown_filename,
        mime="text/markdown",
    )

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write(
        """
        Upload your interview transcripts (one per participant) and specify the desired output file name to perform IPA using ChatGPT.
        """
    )

    uploaded_files = st.file_uploader(
        "Choose transcript text files (one per participant)",
        type=["txt"],
        accept_multiple_files=True,
    )
    output_filename = st.text_input(
        "Enter the desired output file name without extension (e.g., output_analysis)"
    )

    if st.button("Run IPA Analysis"):
        if uploaded_files and output_filename:
            # Ensure output_filename does not have an extension
            output_filename = os.path.splitext(output_filename)[0]
            ipa_analysis_pipeline(uploaded_files, output_filename)
        else:
            st.warning("Please upload transcript files and specify an output file name.")

if __name__ == "__main__":
    main()
