import streamlit as st
import json
import time
import os
import logging
from openai import OpenAI, errors

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
        chat_completion = client.chat.completions.create(
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
        logger.info(f"API Response: {chat_completion}")
        return chat_completion.choices[0].message.content.strip()
    except errors.RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            logger.error("Rate limit exceeded.")
            return ""
    except errors.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        return ""

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

def convert_to_markdown(data):
    """Converts the analysis data to Markdown format."""
    markdown = ""

    # Personal Experiential Themes
    pets_dict = data.get("personal_experiential_themes", {})
    if pets_dict:
        markdown += "# Personal Experiential Themes (PETs)\n\n"
        for participant, participant_data in pets_dict.items():
            markdown += f"## Participant {participant}\n\n"
            pets = participant_data.get("personal_experiential_themes", [])
            if not pets:
                markdown += "No Personal Experiential Themes found.\n\n"
                continue
            for pet in pets:
                pet_title = pet.get('personal_experiential_theme', 'Untitled Theme')
                markdown += f"### {pet_title}\n\n"
                description = pet.get('description', 'N/A')
                markdown += f"**Description:** {description}\n\n"

                extracts = pet.get("extracts", [])
                markdown += f"**Extracts:**\n"
                if extracts:
                    for extract in extracts:
                        markdown += f"- {extract} ({participant})\n"
                else:
                    markdown += "- No extracts provided.\n"

                analytic_comments = pet.get("analytic_comments", [])
                markdown += "\n**Analytic Comments:**\n"
                if analytic_comments:
                    for comment in analytic_comments:
                        markdown += f"- {comment}\n"
                else:
                    markdown += "- No analytic comments provided.\n"

                markdown += "\n"

    # Group Experiential Themes
    gets = data.get("group_experiential_themes", {}).get("group_experiential_themes", [])
    if gets:
        markdown += "# Group Experiential Themes (GETs)\n\n"
        for get in gets:
            get_title = get.get('group_experiential_theme', 'Untitled GET')
            markdown += f"## {get_title}\n\n"
            description = get.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"

            subthemes = get.get('subthemes', [])
            for subtheme in subthemes:
                subtheme_title = subtheme.get('subtheme', 'Untitled Subtheme')
                markdown += f"### {subtheme_title}\n\n"
                sub_description = subtheme.get('description', 'N/A')
                markdown += f"**Description:** {sub_description}\n\n"

                participant_contributions = subtheme.get('participant_contributions', [])
                for contribution in participant_contributions:
                    participant_id = contribution.get('participant_id', 'Unknown Participant')
                    pet_title = contribution.get('pet_title', 'Unknown PET')
                    markdown += f"**{participant_id} - {pet_title}**\n"
                    extracts = contribution.get('extracts', [])
                    if extracts:
                        for extract in extracts:
                            markdown += f"- {extract} ({participant_id})\n"
                    else:
                        markdown += "- No extracts provided.\n"
                    markdown += "\n"

    else:
        markdown += "No Group Experiential Themes found.\n\n"

    return markdown

def ipa_analysis_pipeline(transcripts):
    """Runs the full IPA analysis pipeline on the given transcripts."""
    participant_transcripts = {}
    for idx, transcript_file in enumerate(transcripts):
        try:
            transcript_text = transcript_file.read().decode("utf-8")
            if not transcript_text.strip():
                st.error(f"The uploaded transcript file {transcript_file.name} is empty.")
                return
            participant_transcripts[f"P{idx+1}"] = transcript_text
        except Exception as e:
            st.error(f"Error reading the transcript file {transcript_file.name}: {e}")
            logger.error(f"Error reading the transcript file {transcript_file.name}: {e}")
            return

    # Display transcripts
    st.write("### Uploaded Transcripts:")
    for participant, transcript in participant_transcripts.items():
        with st.expander(f"Transcript {participant}"):
            st.text(transcript)

    # Now, perform the IPA analysis
    # For each participant, generate PETs
    pets_dict = {}
    for participant, transcript in participant_transcripts.items():
        st.write(f"### Analyzing Transcript {participant}")
        with st.spinner(f"Generating PETs for {participant}..."):
            pets_json = generate_pets(transcript, participant)
        if pets_json:
            try:
                pets = json.loads(pets_json)
                pets_dict[participant] = pets
                st.success(f"PETs for {participant} generated successfully.")
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON for PETs of {participant}. Please check the API response.")
                logger.error(f"Error parsing JSON for PETs of {participant}: {e}")
                st.write("API Response Content:")
                st.code(pets_json)
                return
        else:
            st.error(f"Failed to generate PETs for {participant}.")
            return

    # Generate GETs from the PETs
    st.write("### Generating Group Experiential Themes (GETs)...")
    with st.spinner("Generating GETs..."):
        get_json = generate_gets(pets_dict)
    if get_json:
        try:
            gets = json.loads(get_json)
            st.success("GETs generated successfully.")
        except json.JSONDecodeError as e:
            st.error("Error parsing JSON for GETs. Please check the API response.")
            logger.error(f"Error parsing JSON for GETs: {e}")
            st.write("API Response Content:")
            st.code(get_json)
            return
    else:
        st.error("Failed to generate GETs.")
        return

    # Combine PETs and GETs into final analysis
    final_analysis = {
        "personal_experiential_themes": pets_dict,
        "group_experiential_themes": gets
    }

    # Display the final analysis
    st.write("### Final Analysis:")
    markdown_content = convert_to_markdown(final_analysis)
    if not markdown_content:
        st.error("Failed to generate Markdown content.")
        return
    st.markdown(markdown_content)

    # Provide download button for markdown
    st.write("### Download Results:")
    # Use a default filename
    markdown_filename = f"IPA_Analysis.md"

    # Encode the contents
    markdown_bytes = markdown_content.encode("utf-8")

    # Create download button
    st.download_button(
        label="Download Analysis as Markdown",
        data=markdown_bytes,
        file_name=markdown_filename,
        mime="text/markdown",
    )

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
