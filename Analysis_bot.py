import streamlit as st
import json
import time
import logging
import re
from openai import OpenAI, OpenAIError, RateLimitError  # Import exceptions

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve OpenAI API key from Streamlit secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)  # Instantiate the OpenAI client with the API key
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "OPENAI_API_KEY" to your secrets.')
    st.stop()

def fix_json(json_string):
    """Attempts to fix common JSON formatting errors in the assistant's response."""
    json_string = re.sub(r'^[^{]*', '', json_string)  # Remove any text before the first '{'
    json_string = re.sub(r'[^}]*$', '', json_string)  # Remove any text after the last '}'
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)  # Remove trailing commas
    json_string = json_string.replace("'", '"')  # Replace single quotes with double quotes
    json_string = re.sub(r',\s*,', ',', json_string)  # Remove extra commas
    return json_string

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.0, retries=2):
    """Calls the OpenAI API and parses the JSON response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]
        )
        # Ensure content is not empty before parsing
        content = response.choices[0].message.content
        if content:
            return json.loads(fix_json(content))
        else:
            st.error("Received empty response from OpenAI.")
            return {}

    except RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded.")
            return {}
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"JSON decode error: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return {}

def ipa_analysis_pipeline(transcript, output_path):
    """Runs the full IPA analysis pipeline on a given transcript."""
    try:
        transcript_text = transcript.read().decode("utf-8").strip()
        if not transcript_text:
            st.error("The uploaded transcript is empty.")
            return
    except Exception as e:
        st.error(f"Error reading the transcript file: {e}")
        logger.error(f"Error reading the transcript file: {e}")
        return
    
    st.write("### Stage 1: Generating Initial Notes...")
    with st.spinner("Generating initial notes..."):
        initial_notes = stage1_initial_notes(transcript_text)
    
    if not initial_notes:
        st.error("Stage 1 failed. Analysis incomplete.")
        return

    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    with st.spinner("Extracting ES..."):
        es = stage2_experiential_statements(initial_notes)
    
    if not es:
        st.error("Stage 2 failed. Analysis incomplete.")
        return

    st.write("### Stage 3: Clustering PETs...")
    with st.spinner("Clustering PETs..."):
        pets = stage3_cluster_pet(es)
    
    if not pets:
        st.error("Stage 3 failed. Analysis incomplete.")
        return

    st.write("### Stage 4: Writing up GETs...")
    with st.spinner("Writing up GETs..."):
        get_writeup = stage4_get_writeup(pets, transcript_text)
    
    if get_writeup:
        st.write("### Saving Final Analysis to Markdown...")
        save_output(get_writeup, output_path, format="markdown")
        st.markdown(convert_to_markdown(get_writeup))
    else:
        st.error("Stage 4 failed. Analysis incomplete.")
        
def stage1_initial_notes(transcript_text):
    """Stage 1: Close reading and initial notes."""
    prompt = f"""
Perform Stage 1 of Interpretative Phenomenological Analysis (IPA) on the following interview transcript.

Conduct a close reading, making notes about:
- observations
- reflections
- notes
- language use
- context
- interpretative comments
- distinctive phrases
- emotional responses
- reflexivity comments

**Instructions:**
- Provide the output in **strictly valid JSON format**.
- Ensure all necessary commas and syntax are included.
- Do **not** include any additional text, comments, or explanations—only the JSON object.
- Double-check your JSON for correctness before outputting.

Transcript:
{transcript_text}
"""
    result = call_chatgpt(prompt)
    return result if result else {}

def stage2_experiential_statements(initial_notes):
    """Stage 2: Transforming notes into experiential statements."""
    prompt = f"""
    Using the following initial notes from an IPA analysis, transform them into experiential statements.
    Initial Notes:
    {json.dumps(initial_notes, indent=2)}

    Provide the experiential statements in a JSON array format.
    """
    result = call_chatgpt(prompt)
    return result if result else []

def stage3_cluster_pet(es):
    """Stage 3: Clustering experiential statements into Personal Experiential Themes (PETs)."""
    prompt = f"""
    Cluster the following experiential statements into Personal Experiential Themes (PETs).
    ES: {json.dumps(es, indent=2)}
    Output JSON with hierarchy: personal_experiential_theme -> description.
    """
    result = call_chatgpt(prompt)
    return result if result else {}

def stage4_get_writeup(pets, transcript_text):
    """Stage 4: Writing up themes based on Personal and Group Experiential Themes."""
    prompt = f"""
    Write up the themes based on Personal Experiential Themes (PETs), including extracts and analytic comments.
    PETs: {json.dumps(pets, indent=2)}
    Transcript: {transcript_text}
    Output JSON with hierarchy: group_experiential_theme -> personal_experiential_theme -> description, extracts, analytic_comments.
    """
    result = call_chatgpt(prompt)
    return result if result else {}

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
