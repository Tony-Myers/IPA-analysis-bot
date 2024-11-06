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

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.0, retries=2):
    """Calls the OpenAI API and returns the response as text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA). Please use British English spelling in your responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["}"]
        )

        # Extract the message content as plain text
        content = response.choices[0].message.content
        st.write("Raw content received:", content)
        
        return content  # Return the raw text content directly

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

def ipa_analysis_pipeline(transcript):
    """Runs the full IPA analysis pipeline on a given transcript and returns markdown content."""
    try:
        transcript_text = transcript.read().decode("utf-8").strip()
        if not transcript_text:
            st.error("The uploaded transcript is empty.")
            return ""
    except Exception as e:
        st.error(f"Error reading the transcript file: {e}")
        logger.error(f"Error reading the transcript file: {e}")
        return ""
    
    # Stage 1: Generating Initial Notes
    st.write("### Stage 1: Generating Initial Notes...")
    with st.spinner("Generating initial notes..."):
        initial_notes = call_chatgpt(
            f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript. "
            f"Only use the participant’s responses and exclude any interviewer questions or comments.\n\nTranscript:\n{transcript_text}",
            temperature=0.2
        )
    
    if not initial_notes:
        st.error("Stage 1 failed. Analysis incomplete.")
        return ""

    # Stage 2: Formulating Experiential Statements (ES)
    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    with st.spinner("Extracting ES..."):
        es = call_chatgpt(
            f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the participant’s responses. "
            f"Exclude any interviewer questions or comments.\n\nInitial Notes:\n{initial_notes}",
            temperature=0.3
        )
    
    if not es:
        st.error("Stage 2 failed. Analysis incomplete.")
        return ""

    # Stage 3: Clustering PETs with Direct, Verbatim Quotes and Justifications
    st.write("### Stage 3: Clustering PETs...")
    with st.spinner("Clustering PETs..."):
        pets = call_chatgpt(
            f"Using the following Experiential Statements (ES), cluster them into Personal Experiential Themes (PETs). "
            f"For each PET, provide:\n\n"
            f"- **Theme Name**: A concise, creative title.\n"
            f"- **Participant Quotes**: Include direct, verbatim quotes from participants to support the theme. Use only the participant's exact words as they appear in the transcript, without any interpretation, additions, or paraphrasing such as 'the participant.' Each quote should be enclosed in inverted commas and can be truncated with ellipses if necessary to maintain brevity.\n"
            f"- **Researcher Comments**: Concisely explain the relevance of the theme based on participant responses and how it reflects their experiences.\n\n"
            f"Experiential Statements:\n{es}",
            temperature=0.5
        )
    
    if not pets:
        st.error("Stage 3 failed. Analysis incomplete.")
        return ""

    # Stage 4: Writing up GETs with Verbatim Quotes and Justifications
    st.write("### Stage 4: Writing up GETs...")
    with st.spinner("Writing up GETs..."):
        get_writeup = call_chatgpt(
            f"Based on the following Personal Experiential Themes (PETs), synthesise Group Experiential Themes (GETs) as follows:\n\n"
            f"- **Theme Name**: A group-level theme that captures the collective meaning across PETs.\n"
            f"- **Summary**: A short description of the shared experience reflected in this theme.\n"
            f"- **Justifications**: Explain and justify the theme using only direct participant quotes from PETs, with no additional language or interpretation. Use only the participant's exact words as they appear in the transcript, enclosed in inverted commas. Truncate quotes with ellipses if needed, without summarising.\n\n"
            f"Personal Experiential Themes (PETs):\n{pets}",
            temperature=0.7
        )
    
    if get_writeup:
        # Prepare markdown content
        markdown_content = (
            "# IPA Analysis Report\n\n"
            "## Stage 1: Initial Notes\n\n" + initial_notes + "\n\n"
            "## Stage 2: Experiential Statements\n\n" + es + "\n\n"
            "## Stage 3: Personal Experiential Themes (PETs)\n\n" + pets + "\n\n"
            "## Stage 4: Group Experiential Themes (GETs)\n\n" + get_writeup
        )
        return markdown_content
    else:
        st.error("Stage 4 failed. Analysis incomplete.")
        return ""


def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])

    if st.button("Run IPA Analysis"):
        if uploaded_file:
            markdown_content = ipa_analysis_pipeline(uploaded_file)

            if markdown_content:
                # Show analysis results and provide download button
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
            st.warning("Please upload a transcript file.")

if __name__ == "__main__":
    main()
