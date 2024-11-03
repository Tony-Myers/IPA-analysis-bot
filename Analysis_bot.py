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
    return call_chatgpt(prompt, max_tokens=3000)

def generate_gets(participant_pets):
    """Generates GETs by analyzing PETs across participants."""
    pets_across_participants = []
    for participant in participant_pets:
        participant_id = participant['participant_id']
        pets = participant['pets']['personal_experiential_themes']
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
    return call_chatgpt(prompt, max_tokens=3000)

def convert_analysis_to_markdown(analysis):
    """Converts the analysis data to Markdown format, including PETs and GETs."""
    markdown = ""

    # Process PETs for each participant
    participants = analysis.get('participants', [])
    if participants:
        for participant in participants:
            participant_id = participant['participant_id']
            markdown += f"# Personal Experiential Themes for {participant_id}\n\n"
            pets = participant['pets'].get('personal_experiential_themes', [])
            if not pets:
                markdown += f"No Personal Experiential Themes found for {participant_id}.\n\n"
                continue
            for pet in pets:
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
                    markdown += "- No analytic comments provided.\n"
                markdown += "\n"
    else:
        markdown += "No participant data available.\n\n"

    # Process GETs
    gets = analysis.get('group_experiential_themes', [])
    if gets:
        markdown += "# Group Experiential Themes\n\n"
        for get in gets:
            get_title = get.get('group_experiential_theme', 'Untitled GET')
            markdown += f"## {get_title}\n\n"
            description = get.get('description', 'N/A')
            markdown += f"**Description:** {description}\n\n"
            subthemes = get.get('subthemes', [])
            if subthemes:
                for subtheme in subthemes:
                    subtheme_title = subtheme.get('subtheme', 'Untitled Subtheme')
                    markdown += f"### {subtheme_title}\n\n"
                    sub_description = subtheme.get('description', 'N/A')
                    markdown += f"**Description:** {sub_description}\n\n"
                    participant_contributions = subtheme.get('participant_contributions', [])
                    if participant_contributions:
                        for contribution in participant_contributions:
                            pid = contribution.get('participant_id', 'Unknown Participant')
                            extracts = contribution.get('extracts', [])
                            analytic_comments = contribution.get('analytic_comments', [])
                            markdown += f"**Contribution from {pid}:**\n"
                            markdown += f"- **Extracts:**\n"
                            if extracts:
                                for extract in extracts:
                                    markdown += f"  - {extract}\n"
                            else:
                                markdown += "  - No extracts provided.\n"
                            markdown += f"- **Analytic Comments:**\n"
                            if analytic_comments:
                                for comment in analytic_comments:
                                    markdown += f"  - {comment}\n"
                            else:
                                markdown += "  - No analytic comments provided.\n"
                            markdown += "\n"
                    else:
                        markdown += "No participant contributions provided.\n\n"
            else:
                markdown += "No subthemes provided.\n\n"
    else:
        markdown += "No Group Experiential Themes found.\n\n"

    return markdown

def ipa_analysis_pipeline(transcripts, output_filename):
    """Runs the full IPA analysis pipeline on given transcripts."""
    participant_pets = []
    participant_ids = []

    # Display transcripts with option to hide/show
    st.write("### Uploaded Transcripts:")
    for idx, transcript in enumerate(transcripts):
        try:
            transcript_text = transcript.read().decode("utf-8")
            if not transcript_text.strip():
                st.error(f"The uploaded transcript file {transcript.name} is empty.")
                return
            participant_id = f"P{idx+1}"
            participant_ids.append(participant_id)

            with st.expander(f"View Transcript {participant_id} ({transcript.name})"):
                st.text(transcript_text)

            # Process each transcript to generate PETs
            st.write(f"### Processing Transcript {participant_id}...")
            with st.spinner(f"Generating PETs for {participant_id}..."):
                # Call function to generate PETs for this participant
                pets_json = generate_pets(transcript_text, participant_id)
                if pets_json:
                    try:
                        pets = json.loads(pets_json)
                        participant_pets.append({'participant_id': participant_id, 'pets': pets})
                        st.success(f"PETs for {participant_id} generated successfully.")
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON for {participant_id}. Please check the API response.")
                        logger.error(f"Error parsing JSON for {participant_id}: {e}")
                        st.write("API Response Content:")
                        st.code(pets_json)
                else:
                    st.error(f"Failed to generate PETs for {participant_id}.")

        except Exception as e:
            st.error(f"Error reading the transcript file {transcript.name}: {e}")
            logger.error(f"Error reading the transcript file {transcript.name}: {e}")
            return

    # After processing all transcripts, generate GETs
    st.write("### Generating Group Experiential Themes (GETs)...")
    with st.spinner("Generating GETs..."):
        get_json = generate_gets(participant_pets)
        if get_json:
            try:
                gets = json.loads(get_json)
                st.success("GETs generated successfully.")
            except json.JSONDecodeError as e:
                st.error("Error parsing JSON for GETs. Please check the API response.")
                logger.error(f"Error parsing JSON for GETs: {e}")
                st.write("API Response Content:")
                st.code(get_json)
                gets = {}
        else:
            gets = {}

    # Combine PETs and GETs for the final analysis
    analysis = {
        'participants': participant_pets,
        'group_experiential_themes': gets.get('group_experiential_themes', [])
    }

    # Display the analysis
    st.write("### Final Analysis:")
    markdown_content = convert_analysis_to_markdown(analysis)
    if not markdown_content:
        st.error("Failed to generate Markdown content.")
        return
    st.markdown(markdown_content)

    # Provide download button for Markdown
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
        Upload your interview transcripts and specify the desired output file name to perform IPA using ChatGPT.
        """
    )

    uploaded_files = st.file_uploader("Choose transcript text files", type=["txt"], accept_multiple_files=True)
    output_filename = st.text_input("Enter the desired output file name without extension (e.g., output_analysis)")

    if st.button("Run IPA Analysis"):
        if uploaded_files and output_filename:
            # Ensure output_filename does not have an extension
            output_filename = os.path.splitext(output_filename)[0]
            ipa_analysis_pipeline(uploaded_files, output_filename)
        else:
            st.warning("Please upload at least one transcript file and specify an output file name.")

if __name__ == "__main__":
    main()
