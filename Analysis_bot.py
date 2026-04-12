import streamlit as st
import time
import logging
from openai import OpenAI, OpenAIError, RateLimitError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Password Gate ---
def check_password():
    """Returns True if the user has entered the correct password."""
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except KeyError:
        st.error('APP_PASSWORD not found in secrets. Please add it to your secrets.')
        st.stop()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    password_input = st.text_input("Enter the application password:", type="password")
    if password_input:
        if password_input == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# --- DeepSeek Client ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
except KeyError:
    st.error('DeepSeek API key not found in secrets. Please add "DEEPSEEK_API_KEY" to your secrets.')
    st.stop()


def build_system_prompt(reflexive_statement=""):
    """Constructs the system prompt, optionally incorporating the researcher's reflexive statement."""
    base = (
        "You are an expert qualitative researcher specialising in Interpretative Phenomenological Analysis (IPA). "
        "Please use British English spelling in all responses, including quotes."
    )
    if reflexive_statement.strip():
        base += (
            "\n\nThe researcher has provided the following reflexive statement. Use it to inform your "
            "interpretative lens when analysing transcripts — be sensitive to how the researcher's position, "
            "assumptions, and experiences may shape meaning-making:\n\n"
            f"{reflexive_statement}"
        )
    return base


def call_deepseek(prompt, system_prompt, model="deepseek-chat", max_tokens=4096, temperature=0.0, retries=2):
    """Calls the DeepSeek API (OpenAI-compatible) and returns the response as text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = response.choices[0].message.content
        return content
    except RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_deepseek(prompt, system_prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded.")
            return ""
    except OpenAIError as e:
        st.error(f"DeepSeek API error: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""


def analyze_transcript(transcript_text, research_question, aspect, transcript_index, system_prompt):
    """Processes a single transcript to generate Initial Notes, ES, and PETs for a specific aspect."""
    st.write(f"### Processing {aspect} — Stage 1: Generating Initial Notes...")
    initial_notes = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript focusing on '{aspect}' only. "
        f"Use British English spelling.\n\nTranscript:\n{transcript_text}",
        system_prompt=system_prompt,
        temperature=0.2
    )
    if not initial_notes.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty Initial Notes for aspect: {aspect}. Skipping.")
        return None, None, None

    st.write(f"### Processing {aspect} — Stage 2: Formulating Experiential Statements (ES)...")
    es = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the "
        f"participant's responses about '{aspect}'. Use British English spelling.\n\nInitial Notes:\n{initial_notes}",
        system_prompt=system_prompt,
        temperature=0.3
    )
    if not es.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty ES for aspect: {aspect}. Skipping.")
        return None, None, None

    st.write(f"### Processing {aspect} — Stage 3: Clustering PETs...")
    pets = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Using the following Experiential Statements (ES) related to '{aspect}', cluster them into "
        f"Personal Experiential Themes (PETs).\n\nExperiential Statements:\n{es}",
        system_prompt=system_prompt,
        temperature=0.5
    )
    if not pets.strip():
        st.warning(f"Transcript {transcript_index + 1} produced empty PETs for aspect: {aspect}. Skipping.")
        return None, None, None

    return initial_notes, es, pets


def generate_gets(combined_pets, research_question, aspect, system_prompt):
    """Generates Group Experiential Themes (GETs) based on combined PETs for a specific aspect."""
    if not combined_pets.strip():
        st.warning(f"No PETs available to generate GETs for aspect: {aspect}.")
        return "No GETs generated due to lack of PETs."

    st.write(f"### Stage 4: Writing up GETs for {aspect}...")
    get_writeup = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following combined Personal Experiential Themes (PETs) for '{aspect}', "
        f"synthesise Group Experiential Themes (GETs).\n\n"
        f"Combined Personal Experiential Themes (PETs):\n{combined_pets}",
        system_prompt=system_prompt,
        temperature=0.7
    )
    if not get_writeup.strip():
        st.warning(f"Failed to generate GETs for aspect: {aspect}.")
        return "GETs generation failed."

    return get_writeup


def ipa_analysis_pipeline(transcripts, research_question, aspects, system_prompt):
    """Runs the full IPA analysis pipeline on multiple transcripts for each aspect."""
    analysis_results = {}
    markdown_content = ""

    for aspect in aspects:
        all_initial_notes = []
        all_es = []
        all_pets = []

        for i, transcript in enumerate(transcripts):
            try:
                try:
                    transcript_text = transcript.read().decode("utf-8").strip()
                except UnicodeDecodeError:
                    transcript.seek(0)
                    transcript_text = transcript.read().decode("ISO-8859-1").strip()

                if not transcript_text:
                    st.error(f"The uploaded transcript {i + 1} is empty.")
                    continue
            except Exception as e:
                st.error(f"Error reading transcript {i + 1}: {e}")
                logger.error(f"Error reading transcript {i + 1}: {e}")
                continue

            # Reset file pointer for next aspect iteration
            transcript.seek(0)

            st.write(f"## Processing Transcript {i + 1} for Aspect: {aspect}")
            initial_notes, es, pets = analyze_transcript(
                transcript_text, research_question, aspect, i, system_prompt
            )
            if initial_notes and es and pets:
                all_initial_notes.append(initial_notes)
                all_es.append(es)
                all_pets.append(pets)

        combined_pets = "\n\n".join(all_pets)
        get_writeup = generate_gets(combined_pets, research_question, aspect, system_prompt)

        analysis_results[aspect] = {
            "initial_notes": all_initial_notes,
            "es": all_es,
            "pets": all_pets,
            "get_writeup": get_writeup
        }

        markdown_content += f"# Aspect: {aspect}\n\n"
        for i, (initial_notes, es, pets) in enumerate(zip(all_initial_notes, all_es, all_pets)):
            markdown_content += (
                f"## Transcript {i + 1}\n\n"
                f"### Stage 1: Initial Notes\n\n{initial_notes}\n\n"
                f"### Stage 2: Experiential Statements\n\n{es}\n\n"
                f"### Stage 3: Personal Experiential Themes (PETs)\n\n{pets}\n\n"
            )
        markdown_content += f"## Stage 4: Group Experiential Themes (GETs) for {aspect}\n\n{get_writeup}\n\n"

    return markdown_content


def main():
    st.title("IPA Analysis Tool (DeepSeek)")

    if not check_password():
        st.stop()

    research_question = st.text_input("Enter the research question to guide the analysis", "")

    aspects_input = st.text_input("Enter aspects of the research question (comma-separated)", "")
    aspects = [aspect.strip() for aspect in aspects_input.split(",") if aspect.strip()]

    # --- Reflexive Statement ---
    st.subheader("Researcher Reflexive Statement")
    reflexive_file = st.file_uploader(
        "Upload a reflexive statement (.txt)", type=["txt"], key="reflexive"
    )
    reflexive_text_input = st.text_area(
        "Or paste your reflexive statement here:", height=150
    )

    reflexive_statement = ""
    if reflexive_file is not None:
        try:
            reflexive_statement = reflexive_file.read().decode("utf-8").strip()
        except UnicodeDecodeError:
            reflexive_file.seek(0)
            reflexive_statement = reflexive_file.read().decode("ISO-8859-1").strip()
    elif reflexive_text_input.strip():
        reflexive_statement = reflexive_text_input.strip()

    if reflexive_statement:
        st.success("Reflexive statement loaded — it will be used to inform the analysis.")

    # --- Transcript Upload ---
    uploaded_files = st.file_uploader(
        "Choose transcript text files", type=["txt"], accept_multiple_files=True, key="transcripts"
    )

    if st.button("Run IPA Analysis"):
        if research_question and aspects and uploaded_files:
            system_prompt = build_system_prompt(reflexive_statement)
            markdown_content = ipa_analysis_pipeline(
                uploaded_files, research_question, aspects, system_prompt
            )
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
