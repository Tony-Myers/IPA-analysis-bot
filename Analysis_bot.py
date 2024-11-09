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
    client = OpenAI(api_key=api_key)  # Instantiate the OpenAI client with the API key
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

def analyze_transcript(transcript_text):
    """Processes a single transcript to generate Initial Notes, ES, and PETs."""
    st.write("### Stage 1: Generating Initial Notes...")
    initial_notes = call_chatgpt(
        f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript. "
        f"Only use the participant’s responses and exclude any interviewer questions or comments. Use British English spelling throughout.\n\nTranscript:\n{transcript_text}",
        temperature=0.2
    )
    
    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    es = call_chatgpt(
        f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the participant’s responses. "
        f"Exclude any interviewer questions or comments. Use British English spelling in all statements.\n\nInitial Notes:\n{initial_notes}",
        temperature=0.3
    )
    
    st.write("### Stage 3: Clustering PETs...")
    pets = call_chatgpt(
        f"Using the following Experiential Statements (ES), cluster them into Personal Experiential Themes (PETs). "
        f"For each PET, provide:\n\n"
        f"- **Theme Name**: A concise, creative title.\n"
        f"- **Participant Quotes**: Use only direct, verbatim quotes from the transcript with no summary, interpretation, or added language. "
        f"Do not use phrases like 'the participant.' Only the exact words spoken by the participant should be quoted, as they appear in the transcript, with inverted commas around each quote. "
        f"Use British English spelling. If a quote needs truncation, use ellipses without altering the participant's wording.\n"
        f"- **Researcher Comments**: Briefly explain the relevance of the theme based solely on the quotes provided, without rephrasing or summarising the participant's words.\n\n"
        f"Experiential Statements:\n{es}",
        temperature=0.5
    )
    
    return initial_notes, es, pets

def generate_gets(combined_pets):
    """Generates Group Experiential Themes (GETs) based on combined PETs from multiple transcripts."""
    st.write("### Stage 4: Writing up GETs...")
    get_writeup = call_chatgpt(
        f"Based on the following combined Personal Experiential Themes (PETs) from multiple participants, synthesise Group Experiential Themes (GETs) as follows:\n\n"
        f"- **Theme Name**: A group-level theme that captures the collective meaning across PETs.\n"
        f"- **Summary**: A short description of the shared experience reflected in this theme.\n"
        f"- **Justifications**: Use only direct, verbatim participant quotes from the transcript, without rephrasing, summarising, or adding interpretation. "
        f"Each quote should be enclosed in inverted commas, with ellipses if needed to shorten. Avoid any added language, including phrases like 'the participant.' Use British English spelling.\n\n"
        f"Combined Personal Experiential Themes (PETs):\n{combined_pets}",
        temperature=0.7
    )
    return get_writeup

def ipa_analysis_pipeline(transcripts):
    """Runs the full IPA analysis pipeline on multiple transcripts and returns markdown content."""
    all_initial_notes = []
    all_es = []
    all_pets = []

    # Process each transcript individually
    for i, transcript in enumerate(transcripts):
        try:
            # Try to read the transcript with UTF-8 encoding
            try:
                transcript_text = transcript.read().decode("utf-8").strip()
            except UnicodeDecodeError:
                # Fallback to ISO-8859-1 if UTF-8 decoding fails
                transcript.seek(0)  # Reset file pointer to the beginning
                transcript_text = transcript.read().decode("ISO-8859-1").strip()

            if not transcript_text:
                st.error(f"The uploaded transcript {i+1} is empty.")
                return ""
        except Exception as e:
            st.error(f"Error reading transcript {i+1}: {e}")
            logger.error(f"Error reading transcript {i+1}: {e}")
            return ""

        st.write(f"## Processing Transcript {i+1}")
        initial_notes, es, pets = analyze_transcript(transcript_text)
        all_initial_notes.append(initial_notes)
        all_es.append(es)
        all_pets.append(pets)

    # Combine all PETs for GET analysis
    combined_pets = "\n\n".join(all_pets)
    get_writeup = generate_gets(combined_pets)

    # Prepare markdown content for the report
    markdown_content = "# IPA Analysis Report\n\n"
    for i, (initial_notes, es, pets) in enumerate(zip(all_initial_notes, all_es, all_pets)):
        markdown_content += (
            f"## Transcript {i+1}\n\n"
            f"### Stage 1: Initial Notes\n\n{initial_notes}\n\n"
            f"### Stage 2: Experiential Statements\n\n{es}\n\n"
            f"### Stage 3: Personal Experiential Themes (PETs)\n\n{pets}\n\n"
        )
    markdown_content += "## Stage 4: Group Experiential Themes (GETs)\n\n" + get_writeup
    return markdown_content


def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    uploaded_files = st.file_uploader("Choose transcript text files", type=["txt"], accept_multiple_files=True)

    if st.button("Run IPA Analysis"):
        if uploaded_files:
            markdown_content = ipa_analysis_pipeline(uploaded_files)
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
        else:
            st.warning("Please upload at least one transcript file.")

if __name__ == "__main__":
    main()
    
