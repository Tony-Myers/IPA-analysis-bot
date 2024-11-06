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
                {"role": "system", "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA)."},
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
            f"Perform Stage 1 of IPA analysis on the transcript:\n\n{transcript_text}",
            temperature=0.1  # Low temperature for deterministic results
        )
    
    if not initial_notes:
        st.error("Stage 1 failed. Analysis incomplete.")
        return ""

    # Stage 2: Formulating Experiential Statements (ES)
    st.write("### Stage 2: Formulating Experiential Statements (ES)...")
    with st.spinner("Extracting ES..."):
        es = call_chatgpt(
            f"Formulate standardized Experiential Statements (ES) from the following notes:\n\n{initial_notes}\n"
            "Format each ES as a numbered list, clearly stating the experiential theme.",
            temperature=0.3
        )
    
    if not es:
        st.error("Stage 2 failed. Analysis incomplete.")
        return ""

    # Stage 3: Clustering PETs with Quotes and Justifications
    st.write("### Stage 3: Clustering PETs...")
    with st.spinner("Clustering PETs..."):
        pets = call_chatgpt(
            f"Cluster the following ES into PETs (Personal Experiential Themes), each with the following structure:\n\n"
            "- Theme Name: (Creative title with moderate temperature)\n"
            "- Participant Quotes: Short verbatim quotes from the transcript to justify each PET\n"
            "- Researcher Comments: Summarize the significance and recurring elements of the theme\n\n"
            "Use quotes from the relevant participant to enhance each theme justification.",
            temperature=0.5  # Medium temperature to foster creativity in theme naming
        )
    
    if not pets:
        st.error("Stage 3 failed. Analysis incomplete.")
        return ""

    # Stage 4: Writing up GETs with Justifications Based on PETs
    st.write("### Stage 4: Writing up GETs...")
    with st.spinner("Writing up GETs..."):
        get_writeup = call_chatgpt(
            f"Based on the following PETs, write Group Experiential Themes (GETs) as abstract group-level themes.\n\n"
            "Each GET should include:\n"
            "- Theme Name: (Creative group theme name)\n"
            "- Summary: Abstract description of the theme that synthesizes PETs\n"
            "- Justifications: Supporting comments and analysis linking to the PETs\n\n",
            temperature=0.8  # Higher temperature for creative synthesis
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
