import streamlit as st
import time
import logging
from openai import OpenAI, OpenAIError, RateLimitError

# Initialise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Authentication Function ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Safety check: Ensure the secret exists before attempting to read it
    if "APP_PASSWORD" not in st.secrets:
        st.error('App password not configured. Please add "APP_PASSWORD" to your secrets.')
        st.stop()

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Remove password from session state for security
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Please enter the password to access this application", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input("Please enter the password to access this application", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password.")
        return False
    else:
        # Password correct.
        return True

# --- API Initialisation ---
@st.cache_resource
def get_openai_client():
    """Initialises and caches the DeepSeek client securely within the active Streamlit session."""
    if "DEEPSEEK_API_KEY" not in st.secrets:
        st.error('DeepSeek API key not found in secrets. Please add "DEEPSEEK_API_KEY" to your secrets.')
        st.stop()
        
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# --- Core Functions ---
def call_deepseek(prompt, reflexive_statement="", model="deepseek-chat", max_tokens=500, temperature=0.0, retries=2):
    """Calls the DeepSeek API, applying the reflexive statement if provided."""
    
    # Instantiate the client securely within the function scope
    client = get_openai_client()
    
    system_instruction = "You are an expert qualitative researcher specialising in Interpretative Phenomenological Analysis (IPA). Please use British English spelling in all responses, including quotes."
    
    if reflexive_statement.strip():
        system_instruction += f"\n\nCRITICAL CONTEXT - RESEARCHER REFLEXIVITY: When interpreting the data, you must apply the following reflexive statement to inform and contextualise your analysis, acknowledging the researcher's background and perspective:\n{reflexive_statement}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
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
            return call_deepseek(prompt, reflexive_statement, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded.")
            return ""
    except OpenAIError as e:
        st.error(f"API error: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""

def analyze_transcript(transcript_text, research_question, aspect, transcript_index, reflexive_statement):
    """Processes a single transcript to generate Initial Notes, ES, and PETs for a specific aspect."""
    st.write(f"### Processing {aspect} - Stage 1: Generating Initial Notes...")
    initial_notes = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript focusing on '{aspect}' only. "
        f"Use British English spelling.\n\nTranscript:\n{transcript_text}",
        reflexive_statement=reflexive_statement,
        temperature=0.2
    )
    if not initial_notes.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty Initial Notes for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    st.write(f"### Processing {aspect} - Stage 2: Formulating Experiential Statements (ES)...")
    es = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the participant’s responses about '{aspect}'. "
        f"Use British English spelling.\n\nInitial Notes:\n{initial_notes}",
        reflexive_statement=reflexive_statement,
        temperature=0.3
    )
    if not es.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty Experiential Statements for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    st.write(f"### Processing {aspect} - Stage 3: Clustering PETs...")
    pets = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Using the following Experiential Statements (ES) related to '{aspect}', cluster them into Personal Experiential Themes (PETs).\n\n"
        f"Experiential Statements:\n{es}",
        reflexive_statement=reflexive_statement,
        temperature=0.5
    )
    if not pets.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty PETs for aspect: {aspect}. Skipping this transcript.")
        return None, None, None
    
    return initial_notes, es, pets

def generate_gets(combined_pets, research_question, aspect, reflexive_statement):
    """Generates Group Experiential Themes (GETs) based on combined PETs for a specific aspect."""
    if not combined_pets.strip():
        st.warning(f"No PETs available to generate GETs for aspect: {aspect}.")
        return "No GETs generated due to lack of PETs."
    
    st.write(f"### Stage 4: Writing up GETs for {aspect}...")
    get_writeup = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following combined Personal Experiential Themes (PETs) for '{aspect}', synthesise Group Experiential Themes (GETs).\n\n"
        f"Combined Personal Experiential Themes (PETs):\n{combined_pets}",
        reflexive_statement=reflexive_statement,
        temperature=0.7
    )
    if not get_writeup.strip():
        st.warning(f"Failed to generate GETs for aspect: {aspect}.")
        return "GETs generation failed."
    
    return get_writeup

def ipa_analysis_pipeline(transcripts, research_question, aspects, reflexive_statement):
    """Runs the full IPA analysis pipeline on multiple transcripts for each aspect of the research question."""
    analysis_results = {}
    markdown_content = "" 

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
            initial_notes, es, pets = analyze_transcript(transcript_text, research_question, aspect, i, reflexive_statement)
            if initial_notes and es and pets:
                all_initial_notes.append(initial_notes)
                all_es.append(es)
                all_pets.append(pets)

        combined_pets = "\n\n".join(all_pets)
        get_writeup = generate_gets(combined_pets, research_question, aspect, reflexive_statement)

        # Store the results in a dictionary for each aspect
        analysis_results[aspect] = {
            "initial_notes": all_initial_notes,
            "es": all_es,
            "pets": all_pets,
            "get_writeup": get_writeup
        }

        # Append results to markdown_content
        markdown_content += f"# Aspect: {aspect}\n\n"
        for i, (initial_notes, es, pets) in enumerate(zip(all_initial_notes, all_es, all_pets)):
            markdown_content += (
                f"## Transcript {i+1}\n\n"
                f"### Stage 1: Initial Notes\n\n{initial_notes}\n\n"
                f"### Stage 2: Experiential Statements\n\n{es}\n\n"
                f"### Stage 3: Personal Experiential Themes (PETs)\n\n{pets}\n\n"
            )
        markdown_content += f"## Stage 4: Group Experiential Themes (GETs) for {aspect}\n\n{get_writeup}\n\n"

    return markdown_content

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    # Halt execution here if the password is not correct
    if not check_password():
        st.stop()

    # The rest of the app only runs if authentication passes
    research_question = st.text_input("Enter the research question to guide the analysis", "")
    
    aspects_input = st.text_input("Enter aspects of the research question (comma-separated)", "")
    aspects = [aspect.strip() for aspect in aspects_input.split(",") if aspect.strip()]

    # Added text area for Reflexive Statement
    reflexive_statement = st.text_area(
        "Reflexive Statement (Optional)", 
        help="Paste your reflexive statement here. The model will use this to inform its interpretation of the transcripts.",
        height=150
    )
    
    uploaded_files = st.file_uploader("Choose transcript text files", type=["txt"], accept_multiple_files=True)

    if st.button("Run IPA Analysis"):
        if research_question and aspects and uploaded_files:
            markdown_content = ipa_analysis_pipeline(uploaded_files, research_question, aspects, reflexive_statement)
            if markdown_content:
                st.write("### Analysis Complete. Download the Report Below:")
                st.download_button(
                    label="Download Analysis Report",
                    data=markdown_content,
                    file_name="IPA_Analysis_Report.md",
                    mime="text/markdown"
                )
                st.markdown("### Report Preview:")
                st.markdown(markdown_content)
        elif not research_question:
            st.warning("Please enter a research question to direct the analysis.")
        elif not aspects:
            st.warning("Please enter at least one aspect of the research question.")
        else:
            st.warning("Please upload at least one transcript file.")

if __name__ == "__main__":
    main()
