import streamlit as st
import time
import logging
from openai import OpenAI, OpenAIError, RateLimitError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve OpenAI API key from Streamlit secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except KeyError:
    st.error('OpenAI API key not found in secrets. Please add "OPENAI_API_KEY" to your secrets.')
    st.stop()

def call_chatgpt(prompt, model="gpt-4o", max_tokens=1500, temperature=0.0, retries=2):
    """Calls the OpenAI API and returns the response as text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specialising in Interpretative Phenomenological Analysis (IPA). Please use British English spelling in all responses, including quotes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]
        )
        content = response.choices[0].message.content
        return content
    except RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded.")
            return ""
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""

def analyze_transcript(transcript_text, research_question, aspect, transcript_index):
    """Processes a single transcript to generate Initial Notes, ES, and PETs for a specific aspect."""
    st.write(f"### Processing {aspect} - Stage 1: Generating Initial Notes...")
    initial_notes = call_chatgpt(
        f"Research Question Aspect: {aspect}\n\n"
        f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript focusing on '{aspect}' only. "
        f"Use British English spelling.\n\nTranscript:\n{transcript_text}",
        temperature=0.2
    )
    if not initial_notes.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty Initial Notes for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    st.write(f"### Processing {aspect} - Stage 2: Formulating Experiential Statements (ES)...")
    es = call_chatgpt(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the participant’s responses about '{aspect}'. "
        f"Use British English spelling.\n\nInitial Notes:\n{initial_notes}",
        temperature=0.3
    )
    if not es.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty Experiential Statements for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    st.write(f"### Processing {aspect} - Stage 3: Clustering PETs...")
    pets = call_chatgpt(
        f"Research Question Aspect: {aspect}\n\n"
        f"Using the following Experiential Statements (ES) related to '{aspect}', cluster them into Personal Experiential Themes (PETs).\n\n"
        f"Experiential Statements:\n{es}",
        temperature=0.5
    )
    if not pets.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty PETs for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    return initial_notes, es, pets

def generate_gets(combined_pets, research_question, aspect):
    """Generates Group Experiential Themes (GETs) based on combined PETs for a specific aspect."""
    if not combined_pets.strip():
        st.warning(f"No PETs available to generate GETs for aspect: {aspect}.")
        return "No GETs generated due to lack of PETs."
    
    st.write(f"### Stage 4: Writing up GETs for {aspect}...")
    get_writeup = call_chatgpt(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following combined Personal Experiential Themes (PETs) for '{aspect}', synthesise Group Experiential Themes (GETs).\n\n"
        f"Combined Personal Experiential Themes (PETs):\n{combined_pets}",
        temperature=0.7
    )
    if not get_writeup.strip():
        st.warning(f"Failed to generate GETs for aspect: {aspect}.")
        return "GETs generation failed."
    
    return get_writeup

def ipa_analysis_pipeline(transcripts, research_question, aspects):
    """Runs the full IPA analysis pipeline on multiple transcripts for each aspect of the research question."""
    analysis_results = {}

    for aspect in aspects:
        all_initial_notes = []
        all_es = []
        all_pets = []

        # Process each transcript individually for each aspect
        for i, transcript in enumerate(transcripts):
            try:
                # Try to read the transcript with UTF-8 encoding
                try:
                    transcript_text = transcript.read().decode("utf-8").strip()
                except UnicodeDecodeError:
                    transcript.seek(0)
                    transcript_text = transcript.read().decode("ISO-8859-1").strip()

                if not transcript_text:
                    st.error(f"The uploaded transcript {i+1} is empty.")
                    continue
            except Exception as e:
                st.error(f"Error reading transcript {i+1}: {e}")
                logger.error(f"Error reading transcript {i+1}: {e}")
                continue

            st.write(f"## Processing Transcript {i+1} for Aspect: {aspect}")
            initial_notes, es, pets = analyze_transcript(transcript_text, research_question, aspect, i)
            if initial_notes and es and pets:
                all_initial_notes.append(initial_notes)
                all_es.append(es)
                all_pets.append(pets)

        combined_pets = "\n\n".join(all_pets)
        get_writeup = generate_gets(combined_pets, research_question, aspect)

        # Store the results in a dictionary for each aspect
        analysis_results[aspect] = {
            "initial_notes": all_initial_notes,
            "es": all_es,
            "pets": all_pets,
            "get_writeup": get_writeup
        }

    return analysis_results

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    # Input for research question
    research_question = st.text_input("Enter the research question to
::contentReference[oaicite:0]{index=0}
 
