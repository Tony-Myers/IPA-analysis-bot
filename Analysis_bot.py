import streamlit as st
import openai
import json
import time

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
        time.sleep(60)
        return call_chatgpt(prompt, model, max_tokens, temperature)
    except Exception as e:
        st.error(f"An error occurred: {e}")
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
    Using the following clustered themes from an IPA analysis, write up each theme by describing it, providing relevant extracts from the transcript, and adding analytic comments.

    Clustered Themes:
    {json.dumps(clustered_themes, indent=2)}

    Transcript:
    {transcript}

    Please provide the write-up in a structured JSON format with the following fields for each theme:
    - superordinate_theme
        - subtheme
            - description
            - extracts
            - analytic_comments
    """
    return call_chatgpt(prompt)

def save_output(data, file_path):
    """Saves the data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def ipa_analysis_pipeline(transcript, output_path):
    """Runs the full IPA analysis pipeline on a given transcript."""
    transcript_text = transcript.read().decode("utf-8")
    
    st.write("### Stage 1: Generating Initial Notes...")
    initial_notes_json = stage1_initial_notes(transcript_text)
    try:
        initial_notes = json.loads(initial_notes_json)
        st.success("Stage 1 completed successfully.")
    except json.JSONDecodeError:
        st.error("Error parsing JSON from Stage 1. Please check the API response.")
        initial_notes = {}
    
    st.write("### Stage 2: Extracting Emergent Themes...")
    emergent_themes_json = stage2_emergent_themes(initial_notes)
    try:
        emergent_themes = json.loads(emergent_themes_json)
        st.success("Stage 2 completed successfully.")
    except json.JSONDecodeError:
        st.error("Error parsing JSON from Stage 2. Please check the API response.")
        emergent_themes = []
    
    st.write("### Stage 3: Clustering Themes...")
    clustered_themes_json = stage3_cluster_themes(emergent_themes)
    try:
        clustered_themes = json.loads(clustered_themes_json)
        st.success("Stage 3 completed successfully.")
    except json.JSONDecodeError:
        st.error("Error parsing JSON from Stage 3. Please check the API response.")
        clustered_themes = {}
    
    st.write("### Stage 4: Writing Up Themes with Extracts and Comments...")
    write_up_json = stage4_write_up_themes(clustered_themes, transcript_text)
    try:
        write_up = json.loads(write_up_json)
        st.success("Stage 4 completed successfully.")
    except json.JSONDecodeError:
        st.error("Error parsing JSON from Stage 4. Please check the API response.")
        write_up = {}
    
    if write_up:
        st.write("### Saving the Final Analysis to File...")
        save_output(write_up, output_path)
        st.success(f"IPA analysis complete. Results saved to {output_path}")
        st.json(write_up)  # Display the JSON output in the app

# Streamlit App Layout
def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write("""
    Upload your interview transcript and specify the output JSON file path to perform IPA using ChatGPT.
    """)

    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])
    output_path = st.text_input("Enter the desired output JSON file path (e.g., results/output_analysis.json)")

    if st.button("Run IPA Analysis"):
        if uploaded_file and output_path:
            ipa_analysis_pipeline(uploaded_file, output_path)
        else:
            st.warning("Please upload a transcript file and specify an output path.")

if __name__ == "__main__":
    main()
