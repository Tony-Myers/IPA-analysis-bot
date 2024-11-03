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

def call_chatgpt(:
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
    Includes error handling and rate limiting.
    """
    try:
        response = client.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3,
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
            with open(file_path, 'w', encoding='utf-8') as file:
                markdown_content = convert_to_markdown(data)
                file.write(markdown_content)
        elif format == "text":
            with open(file_path, 'w', encoding='utf-8') as file:
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
    """
    return call_chatgpt(prompt)

def stage2_emergent_themes(initial_notes):
    """Stage 2: Transforming notes into emergent themes."""
    prompt = f"""
    Using the following initial notes from an IPA analysis, transform them into emergent themes.
    Formulate concise phrases at a higher level of abstraction grounded in the participant’s account.

    Initial Notes:
    {json.dumps(initial_notes, indent=2)}

    Provide the emergent themes in a JSON array format.
    """
    return call_chatgpt(prompt)

def stage3_cluster_themes(emergent_themes):
    """Stage 3: Seeking relationships and clustering themes."""
    prompt = f"""
    Based on the following emergent themes from an IPA analysis, identify connections between them,
    group them into clusters based on conceptual similarities, and organize them into superordinate themes and subthemes.

    Emergent Themes:
    {json.dumps(emergent_themes, indent=2)}

    Provide the clustered themes in a structured JSON format with the following hierarchy:
    - superordinate_theme
        - subtheme
    """
    return call_chatgpt(prompt)

def stage4_write_up_themes(clustered_themes, transcript):
    """Stage 4: Writing up themes with extracts and analytic comments."""
    prompt = f"""
    Using the following clustered themes from an IPA analysis, concisely write up each theme.
    For each theme, include a brief description, relevant extracts from the transcript, and analytic comments.

    Clustered Themes:
    {json.dumps(clustered_themes, indent=2)}

    Transcript:
    {transcript}

    Provide the output in a well-formatted JSON structure with these fields for each theme:
    - superordinate_theme
        - subtheme
            - description
            - extracts
            - analytic_comments

    Ensure the JSON is complete and properly formatted.
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
    
    st.write("### Stage 2: Extracting Emergent Themes...")
    with st.spinner("Extracting emergent themes..."):
        emergent_themes_json = stage2_emergent_themes(initial_notes)
    
    if emergent_themes_json:
        try:
            emergent_themes = json.loads(emergent_themes_json)
            st.success("Stage 2 completed successfully.")
        except json.JSONDecodeError:
            st.error("Error parsing JSON from Stage 2. Please check the API response.")
            logger.error("Error parsing JSON from Stage 2. Please check the API response.")
            emergent_themes = []
    else:
        emergent_themes = []
    
    if not emergent_themes:
        st.error("Stage 2 failed. Aborting the pipeline.")
        return
    
    st.write("### Stage 3: Clustering Themes...")
    with st.spinner("Clustering themes..."):
        clustered_themes_json = stage3_cluster_themes(emergent_themes)
    
    if clustered_themes_json:
        try:
            clustered_themes = json.loads(clustered_themes_json)
            st.success("Stage 3 completed successfully.")
        except json.JSONDecodeError:
            st.error("Error parsing JSON from Stage 3. Please check the API response.")
            logger.error("Error parsing JSON from Stage 3. Please check the API response.")
            clustered_themes = {}
    else:
        clustered_themes = {}
    
    if not clustered_themes:
        st.error("Stage 3 failed. Aborting the pipeline.")
        return
    
    st.write("### Stage 4: Writing Up Themes with Extracts and Comments...")
    with st.spinner("Writing up themes..."):
        write_up_json = stage4_write_up_themes(clustered_themes, transcript_text)
    
    if write_up_json:
        try:
            write_up = json.loads(write_up_json)
            st.success("Stage 4 completed successfully.")
        except json.JSONDecodeError:
            st.warning("Error parsing JSON from Stage 4. Retrying with adjusted parameters...")
            logger.warning("Error parsing JSON from Stage 4. Retrying with adjusted parameters.")
            # Retry with further reduced max_tokens
            write_up_json = call_chatgpt(
                prompt=stage4_write_up_themes(clustered_themes, transcript_text),
                model="gpt-4",
                max_tokens=800,  # Further reduced tokens
                temperature=0.3,
                retries=1
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
        save_output(write_up, output_path, format="json")
        st.write("### Final Analysis:")
        st.json(write_up)
        
        # Convert data to desired formats
        markdown_content = convert_to_markdown(write_up)
        text_content = convert_to_text(write_up)
        
        # Provide download buttons
        st.write("### Download Results:")
        
        # Prepare file names
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        json_filename = f"{base_filename}.json"
        markdown_filename = f"{base_filename}.md"
        text_filena
