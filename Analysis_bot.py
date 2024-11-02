import streamlit as st
import openai
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

def call_chatgpt(prompt, model="gpt-4", max_tokens=800, temperature=0.3, retries=2):
    """
    Calls the OpenAI Chat API to get a completion.
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
        logger.info(f"API Response: {response}")
        return response.choices[0].message["content"].strip()
    except openai.error.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""

def convert_to_markdown(data):
    """Converts the analysis data to Markdown format."""
    markdown = ""
    for get in data.get("group_experiential_themes", []):
        markdown += f"## {get['group_experiential_theme']}\n\n"
        for pet in get.get("personal_experiential_themes", []):
            markdown += f"### {pet['personal_experiential_theme']}\n\n"
            markdown += f"**Description:** {pet.get('description', 'N/A')}\n\n"
            markdown += f"**Extracts:**\n"
            for extract in pet.get("extracts", []):
                markdown += f"- {extract}\n"
            markdown += "\n**Analytic Comments:**\n"
            for comment in pet.get("analytic_comments", []):
                markdown += f"- {comment}\n"
            markdown += "\n"
    return markdown

def save_output(data, file_path, format="markdown"):
    """Saves the data to a specified file format."""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        markdown_content = convert_to_markdown(data)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(markdown_content)
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

def stage2_experiential_statements(initial_notes):
    """Stage 2: Transforming notes into experiential statements."""
    prompt = f"""
    Using the following initial notes from an IPA analysis, transform them into experiential statements.
    Initial Notes:
    {json.dumps(initial_notes, indent=2)}

    Provide the experiential statements in a JSON array format.
    """
    return call_chatgpt(prompt)

def stage3_cluster_pet(es):
    """Stage 3: Clustering experiential statements into Personal Experiential Themes (PETs)."""
    prompt = f"""
    Cluster the following experiential statements into Personal Experiential Themes (PETs).
    ES: {json.dumps(es, indent=2)}
    Output JSON with hierarchy: personal_experiential_theme -> description.
    """
    return call_chatgpt(prompt)

def stage4_get_writeup(pets, transcript):
    """Stage 4: Writing up themes based on Personal and Group Experiential Themes."""
    prompt = f"""
    Write up the themes based on Personal Experiential Themes (PETs), including extracts and analytic comments.
    PETs: {json.dumps(pets, indent=2)}
    Transcript: {transcript}
    Output JSON with hierarchy: group_experiential_theme -> personal_experiential_theme -> description, extracts, analytic_comments.
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
        initial_notes = json.loads(initial_notes_json) if initial_notes_json else {}

    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    with st.spinner("Extracting ES..."):
        es_json = stage2_experiential_statements(initial_notes)
        es = json.loads(es_json) if es_json else []

    st.write("### Stage 3: Clustering PETs...")
    with st.spinner("Clustering PETs..."):
        pets_json = stage3_cluster_pet(es)
        pets = json.loads(pets_json) if pets_json else {}

    st.write("### Stage 4: Writing up GETs...")
    with st.spinner("Writing up GETs..."):
        get_writeup_json = stage4_get_writeup(pets, transcript_text)
        get_writeup = json.loads(get_writeup_json) if get_writeup_json else {}

    if get_writeup:
        st.write("### Saving Final Analysis to Markdown...")
        save_output(get_writeup, output_path, format="markdown")
        st.write("### Final Analysis (Markdown Format):")
        st.markdown(convert_to_markdown(get_writeup))
    else:
        st.error("Stage 4 failed. Analysis incomplete.")

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])
    output_path = st.text_input("Enter the desired output file path (e.g., results/output_analysis.md)")

    if st.button("Run IPA Analysis"):
        if uploaded_file and output_path:
            ipa_analysis_pipeline(uploaded_file, output_path)
        else:
            st.warning("Please upload a transcript file and specify an output path.")

if __name__ == "__main__":
    main()
