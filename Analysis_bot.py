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
            with open(file_path, "w", encoding="utf-8") as file:
                text_content = convert_to_text(data)
                file.write(text_content)
        else:
            st.error(f"Unsupported format: {format}")
            return

        st.success(f"IPA analysis complete. Results saved to {file_path}")
    except Exception as e:
        st.error(f"Failed to save the output file: {e}")
        logger.error(f"Failed to save the output file: {e}")

def stage1_initial_notes(transcript):
    """Stage 1: Close reading and initial notes."""
    prompt = f"""
Perform Stage 1 of Interpretative Phenomenological Analysis (IPA) on the following interview transcript.
Conduct a close reading, making notes about observations, reflections, content, language use, context, and initial interpretative comments.
Highlight distinctive phrases and emotional responses. Include any personal reflexivity comments if relevant.

Transcript:
{transcript}

Provide the output in a structured JSON format with the following fields:
- observations
- reflections
- content_notes
- language_use
- context
- interpretative_comments
- distinctive_phrases
- emotional_responses
- reflexivity_comments
Ensure the JSON is complete and properly formatted.
"""
    return call_chatgpt(prompt)

def stage2_experiential_statements(initial_notes):
    """Stage 2: Formulating Experiential Statements (ES)."""
    prompt = f"""
Using the following initial notes from an IPA analysis, formulate Experiential Statements (ES).
Formulate concise phrases at a higher level of abstraction grounded in the participant’s account.

Initial Notes:
{json.dumps(initial_notes, indent=2)}

Provide the Experiential Statements in a JSON array format.
Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON array.
"""
    return call_chatgpt(prompt)

def stage3_personal_experiential_themes(experiential_statements):
    """Stage 3: Clustering Experiential Statements into Personal Experiential Themes (PETs)."""
    prompt = f"""
Based on the following Experiential Statements (ES) from an IPA analysis, identify connections between them,
group them into clusters based on conceptual similarities, and organize them into Personal Experiential Themes (PETs).

Experiential Statements:
{json.dumps(experiential_statements, indent=2)}

Provide the clustered themes in a structured JSON format with the following hierarchy:
- personal_experiential_theme
    - experiential_statements
Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.
"""
    return call_chatgpt(prompt)

def stage4_write_up_pet(pets, transcript):
    """Stage 4: Writing up Personal Experiential Themes (PETs) with extracts and analytic comments."""
    prompt = f"""
Using the following Personal Experiential Themes (PETs) from an IPA analysis, concisely write up each theme.
For each PET, include a brief description, relevant extracts from the transcript, and analytic comments.

Personal Experiential Themes (PETs):
{json.dumps(pets, indent=2)}

Transcript:
{transcript}

Provide the output in a well-formatted JSON structure with these fields for each PET:
- personal_experiential_theme
    - description
    - extracts
    - analytic_comments

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.
"""
    return call_chatgpt(prompt)

def ipa_analysis_pipeline(transcript, output_path):
    """Runs the full IPA analysis pipeline on a given transcript."""
    try:
        transcript_text = transcript.read().decode("utf-8")
        if not transcript_text.strip():
            st.error("The uploaded transcript file is empty.")
            return
    except Exception as e:
        st.error(f"Error reading the transcript file: {e}")
        logger.error(f"Error reading the transcript file: {e}")
        return

    st.write("### Stage 1: Generating Initial Notes...")
    with st.spinner("Generating initial notes..."):
        initial_notes_json = stage1_initial_notes(transcript_text)

    if initial_notes_json:
        try:
            initial_notes = json.loads(initial_notes_json)
            st.success("Stage 1 completed successfully.")
        except json.JSONDecodeError:
            st.error("Error parsing JSON from Stage 1. Please check the API response.")
            logger.error("Error parsing JSON from Stage 1. Please check the API response.")
            initial_notes = {}
    else:
        initial_notes = {}

    if not initial_notes:
        st.error("Stage 1 failed. Aborting the pipeline.")
        return

    st.write("### Stage 2: Formulating Experiential Statements...")
    with st.spinner("Formulating experiential statements..."):
        experiential_statements_json = stage2_experiential_statements(initial_notes)

    if experiential_statements_json:
        try:
            experiential_statements = json.loads(experiential_statements_json)
            st.success("Stage 2 completed successfully.")
        except json.JSONDecodeError:
            st.error("Error parsing JSON from Stage 2. Please check the API response.")
            logger.error("Error parsing JSON from Stage 2. Please check the API response.")
            experiential_statements = []
    else:
        experiential_statements = []

    if not experiential_statements:
        st.error("Stage 2 failed. Aborting the pipeline.")
        return

    st.write("### Stage 3: Clustering into Personal Experiential Themes (PETs)...")
    with st.spinner("Clustering experiential statements..."):
        pets_json = stage3_personal_experiential_themes(experiential_statements)

    if pets_json:
        try:
            pets = json.loads(pets_json)
            st.success("Stage 3 completed successfully.")
        except json.JSONDecodeError:
            st.error("Error parsing JSON from Stage 3. Please check the API response.")
            logger.error("Error parsing JSON from Stage 3. Please check the API response.")
            pets = {}
    else:
        pets = {}

    if not pets:
        st.error("Stage 3 failed. Aborting the pipeline.")
        return

    st.write("### Stage 4: Writing Up PETs with Extracts and Comments...")
    with st.spinner("Writing up PETs..."):
        write_up_json = stage4_write_up_pet(pets, transcript_text)

    if write_up_json:
        try:
            write_up = json.loads(write_up_json)
            st.success("Stage 4 completed successfully.")
        except json.JSONDecodeError:
            st.warning("Error parsing JSON from Stage 4. Retrying with adjusted parameters...")
            logger.warning("Error parsing JSON from Stage 4. Retrying with adjusted parameters.")
            # Retry with further reduced max_tokens
            write_up_json = call_chatgpt(
                prompt=stage4_write_up_pet(pets, transcript_text),
                model="gpt-4",
                max_tokens=500,  # Further reduced tokens
                temperature=0.3,
                retries=1,
            )
            try:
                write_up = json.loads(write_up_json)
                st.success("Stage 4 completed successfully on retry.")
            except json.JSONDecodeError:
                st.error("Error parsing JSON from Stage 4 after retry. Please check the API response.")
                logger.error("Error parsing JSON from Stage 4 after retry. Please check the API response.")
                write_up = {}
    else:
        write_up = {}

    if write_up:
        st.write("### Saving the Final Analysis to File...")
        save_output(write_up, output_path, format="markdown")
        st.write("### Final Analysis:")
        st.markdown(convert_to_markdown(write_up))
        
        # Convert data to desired formats
        markdown_content = convert_to_markdown(write_up)
        text_content = convert_to_text(write_up)

        # Provide download buttons
        st.write("### Download Results:")

        # Prepare file names
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        markdown_filename = f"{base_filename}.md"
        text_filename = f"{base_filename}.txt"
        json_filename = f"{base_filename}.json"

        # Encode the contents
        markdown_bytes = markdown_content.encode("utf-8")
        text_bytes = text_content.encode("utf-8")
        json_bytes = json.dumps(write_up, indent=2).encode("utf-8")

        # Create download buttons
        st.download_button(
            label="Download Markdown",
            data=markdown_bytes,
            file_name=markdown_filename,
            mime="text/markdown",
        )

        st.download_button(
            label="Download Text",
            data=text_bytes,
            file_name=text_filename,
            mime="text/plain",
        )

        # Optionally provide JSON download
        st.download_button(
            label="Download JSON",
            data=json_bytes,
            file_name=json_filename,
            mime="application/json",
        )
    else:
        st.error("Stage 4 failed. Analysis incomplete.")

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write(
        """
    Upload your interview transcript and specify the output file path to perform IPA using ChatGPT.
    """
    )

    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])
    output_path = st.text_input("Enter the desired output file path without extension (e.g., results/output_analysis)")

    if st.button("Run IPA Analysis"):
        if uploaded_file and output_path:
            # Ensure output_path does not have an extension
            output_path = os.path.splitext(output_path)[0]
            ipa_analysis_pipeline(uploaded_file, output_path)
        else:
            st.warning("Please upload a transcript file and specify an output path.")

if __name__ == "__main__":
    main()
