import streamlit as st
import openai
import json
import time
import os
import logging
import re
import openai.error

from openai.error import OpenAIError, RateLimitError
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize OpenAI client with the API key from Streamlit secrets
try:
    api_key = st.secrets["openai_api_key"]
    client = openai.OpenAI(api_key=api_key)
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "openai_api_key" to your secrets.')
    st.stop()

def fix_json(json_string):
    """
    Attempts to fix common JSON formatting errors in the assistant's response.
    """
    # [Contents of fix_json function as provided above]

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.2, retries=2):
    """
    Calls the OpenAI API and parses the JSON response.
    """
    function = {
        "name": "ipa_analysis_stage1",
        "description": "Performs Stage 1 IPA analysis and returns structured notes.",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {"type": "array", "items": {"type": "string"}},
                "reflections": {"type": "array", "items": {"type": "string"}},
                "content_notes": {"type": "array", "items": {"type": "string"}},
                "language_use": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "array", "items": {"type": "string"}},
                "interpretative_comments": {"type": "array", "items": {"type": "string"}},
                "distinctive_phrases": {"type": "array", "items": {"type": "string"}},
                "emotional_responses": {"type": "array", "items": {"type": "string"}},
                "reflexivity_comments": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["observations", "reflections", "content_notes", "language_use",
                         "context", "interpretative_comments", "distinctive_phrases",
                         "emotional_responses", "reflexivity_comments"],
        },
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert qualitative researcher specializing in "
                        "Interpretative Phenomenological Analysis (IPA). "
                        "You output data in valid JSON format without any additional text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            functions=[function],
            function_call={"name": "ipa_analysis_stage1"},
            max_tokens=max_tokens,
            temperature=temperature,
        )

            message = response.choices[0].message
            function_call = getattr(message, "function_call", None)

    if function_call and hasattr(function_call, "arguments"):
            arguments = function_call.arguments
    else:
    arguments = "{}"  # Default to an empty JSON object if no arguments are found

        logger.info(f"Raw API Response: {arguments}")

        parsed_result = json.loads(arguments)
        return parsed_result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        st.error(f"JSON parsing error: {e}")
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
- content
- language use
- context
- initial interpretative comments
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
    return json.loads(result) if result else []

def stage3_cluster_pet(es):
    """Stage 3: Clustering experiential statements into Personal Experiential Themes (PETs)."""
    prompt = f"""
    Cluster the following experiential statements into Personal Experiential Themes (PETs).
    ES: {json.dumps(es, indent=2)}
    Output JSON with hierarchy: personal_experiential_theme -> description.
    """
    result = call_chatgpt(prompt)
    return json.loads(result) if result else {}

def stage4_get_writeup(pets, transcript_text):
    """Stage 4: Writing up themes based on Personal and Group Experiential Themes."""
    prompt = f"""
    Write up the themes based on Personal Experiential Themes (PETs), including extracts and analytic comments.
    PETs: {json.dumps(pets, indent=2)}
    Transcript: {transcript_text}
    Output JSON with hierarchy: group_experiential_theme -> personal_experiential_theme -> description, extracts, analytic_comments.
    """
    result = call_chatgpt(prompt)
    return json.loads(result) if result else {}


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
