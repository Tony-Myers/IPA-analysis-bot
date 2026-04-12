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
def get_client():
    """Returns the DeepSeek client, creating it if necessary."""
    if "deepseek_client" not in st.session_state:
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            st.session_state.deepseek_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        except KeyError:
            st.error('DeepSeek API key not found in secrets. Please add "DEEPSEEK_API_KEY" to your secrets.')
            st.stop()
    return st.session_state.deepseek_client


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
    client = get_client()
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
        logger.info(f"API response received: {len(content)} chars")
        return content if content else ""
    except RateLimitError:
        if retries > 0:
            logger.warning("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return call_deepseek(prompt, system_prompt, model, max_tokens, temperature, retries - 1)
        else:
            logger.error("Rate limit exceeded after all retries.")
            return ""
    except OpenAIError as e:
        logger.error(f"DeepSeek API error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ""


def analyze_transcript(transcript_text, aspect, system_prompt, status):
    """Processes a single transcript to generate Initial Notes, ES, and PETs for a specific aspect."""
    status.update(label=f"Stage 1: Generating Initial Notes for '{aspect}'...")
    initial_notes = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Perform Stage 1 of IPA analysis on the participant's responses in the transcript focusing on '{aspect}' only. "
        f"Use British English spelling.\n\nTranscript:\n{transcript_text}",
        system_prompt=system_prompt,
        temperature=0.2
    )
    if not initial_notes.strip():
        logger.warning(f"Empty Initial Notes for aspect: {aspect}")
        return None, None, None
    status.write(f"✓ Initial Notes generated ({len(initial_notes)} chars)")

    status.update(label=f"Stage 2: Formulating Experiential Statements for '{aspect}'...")
    es = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following initial notes, formulate Experiential Statements (ES) focusing solely on the "
        f"participant's responses about '{aspect}'. Use British English spelling.\n\nInitial Notes:\n{initial_notes}",
        system_prompt=system_prompt,
        temperature=0.3
    )
    if not es.strip():
        logger.warning(f"Empty ES for aspect: {aspect}")
        return None, None, None
    status.write(f"✓ Experiential Statements generated ({len(es)} chars)")

    status.update(label=f"Stage 3: Clustering PETs for '{aspect}'...")
    pets = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Using the following Experiential Statements (ES) related to '{aspect}', cluster them into "
        f"Personal Experiential Themes (PETs).\n\nExperiential Statements:\n{es}",
        system_prompt=system_prompt,
        temperature=0.5
    )
    if not pets.strip():
        logger.warning(f"Empty PETs for aspect: {aspect}")
        return None, None, None
    status.write(f"✓ PETs generated ({len(pets)} chars)")

    return initial_notes, es, pets


def generate_gets(combined_pets, aspect, system_prompt, status):
    """Generates Group Experiential Themes (GETs) based on combined PETs for a specific aspect."""
    if not combined_pets.strip():
        return "No GETs generated due to lack of PETs."

    status.update(label=f"Stage 4: Synthesising GETs for '{aspect}'...")
    get_writeup = call_deepseek(
        f"Research Question Aspect: {aspect}\n\n"
        f"Based on the following combined Personal Experiential Themes (PETs) for '{aspect}', "
        f"synthesise Group Experiential Themes (GETs).\n\n"
        f"Combined Personal Experiential Themes (PETs):\n{combined_pets}",
        system_prompt=system_prompt,
        temperature=0.7
    )
    if not get_writeup.strip():
        return "GETs generation failed."

    status.write(f"✓ GETs generated ({len(get_writeup)} chars)")
    return get_writeup


def ipa_analysis_pipeline(transcript_contents, aspects, system_prompt):
    """Runs the full IPA analysis pipeline on pre-read transcript contents for each aspect."""
    markdown_content = ""

    total_steps = len(aspects) * len(transcript_contents) + len(aspects)  # transcripts + GETs per aspect
    current_step = 0
    progress_bar = st.progress(0, text="Starting analysis...")

    for aspect in aspects:
        all_initial_notes = []
        all_es = []
        all_pets = []

        for i, transcript_text in enumerate(transcript_contents):
            with st.status(f"Transcript {i + 1} — {aspect}", expanded=True) as status:
                initial_notes, es, pets = analyze_transcript(
                    transcript_text, aspect, system_prompt, status
                )
                if initial_notes and es and pets:
                    all_initial_notes.append(initial_notes)
                    all_es.append(es)
                    all_pets.append(pets)
                    status.update(label=f"✓ Transcript {i + 1} — {aspect} complete", state="complete")
                else:
                    status.update(label=f"⚠ Transcript {i + 1} — {aspect} produced empty results", state="error")

            current_step += 1
            progress_bar.progress(current_step / total_steps, text=f"Progress: {current_step}/{total_steps}")

        # Generate GETs for this aspect
        combined_pets = "\n\n".join(all_pets)
        with st.status(f"Generating GETs — {aspect}", expanded=True) as status:
            get_writeup = generate_gets(combined_pets, aspect, system_prompt, status)
            status.update(label=f"✓ GETs for '{aspect}' complete", state="complete")

        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"Progress: {current_step}/{total_steps}")

        # Build markdown for this aspect
        markdown_content += f"# Aspect: {aspect}\n\n"
        for i, (notes, es, pets) in enumerate(zip(all_initial_notes, all_es, all_pets)):
            markdown_content += (
                f"## Transcript {i + 1}\n\n"
                f"### Stage 1: Initial Notes\n\n{notes}\n\n"
                f"### Stage 2: Experiential Statements\n\n{es}\n\n"
                f"### Stage 3: Personal Experiential Themes (PETs)\n\n{pets}\n\n"
            )
        markdown_content += f"## Stage 4: Group Experiential Themes (GETs) for {aspect}\n\n{get_writeup}\n\n"

        # Store partial results after each aspect completes
        st.session_state.analysis_report = markdown_content
        logger.info(f"Aspect '{aspect}' complete. Report so far: {len(markdown_content)} chars")

    progress_bar.progress(1.0, text="Analysis complete!")
    return markdown_content


def read_transcript_contents(uploaded_files):
    """Reads all uploaded files into memory as strings. Returns list of text contents."""
    contents = []
    for i, f in enumerate(uploaded_files):
        try:
            try:
                text = f.read().decode("utf-8").strip()
            except UnicodeDecodeError:
                f.seek(0)
                text = f.read().decode("ISO-8859-1").strip()

            if not text:
                st.error(f"Transcript {i + 1} ({f.name}) is empty.")
                continue
            contents.append(text)
            logger.info(f"Read transcript {i + 1} ({f.name}): {len(text)} chars")
        except Exception as e:
            st.error(f"Error reading transcript {i + 1} ({f.name}): {e}")
            logger.error(f"Error reading transcript {i + 1}: {e}")
    return contents


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
        if ("reflexive_content" not in st.session_state
                or st.session_state.get("reflexive_filename") != reflexive_file.name):
            try:
                reflexive_statement = reflexive_file.read().decode("utf-8").strip()
            except UnicodeDecodeError:
                reflexive_file.seek(0)
                reflexive_statement = reflexive_file.read().decode("ISO-8859-1").strip()
            st.session_state.reflexive_content = reflexive_statement
            st.session_state.reflexive_filename = reflexive_file.name
        else:
            reflexive_statement = st.session_state.reflexive_content
    elif reflexive_text_input.strip():
        reflexive_statement = reflexive_text_input.strip()

    if reflexive_statement:
        st.success("Reflexive statement loaded — it will be used to inform the analysis.")

    # --- Transcript Upload ---
    uploaded_files = st.file_uploader(
        "Choose transcript text files", type=["txt"], accept_multiple_files=True
    )

    # --- Run Analysis ---
    if st.button("Run IPA Analysis"):
        if not research_question:
            st.warning("Please enter a research question to direct the analysis.")
        elif not aspects:
            st.warning("Please enter at least one aspect of the research question.")
        elif not uploaded_files:
            st.warning("Please upload at least one transcript file.")
        else:
            # Read all transcript content into memory BEFORE starting
            transcript_contents = read_transcript_contents(uploaded_files)
            if not transcript_contents:
                st.error("No valid transcript content could be read.")
            else:
                st.info(f"Starting analysis: {len(transcript_contents)} transcript(s), {len(aspects)} aspect(s)")
                system_prompt = build_system_prompt(reflexive_statement)
                markdown_content = ipa_analysis_pipeline(
                    transcript_contents, aspects, system_prompt
                )
                if markdown_content.strip():
                    st.session_state.analysis_report = markdown_content
                    st.session_state.analysis_complete = True
                    logger.info(f"Analysis complete. Report length: {len(markdown_content)} chars")
                else:
                    st.error("Analysis completed but produced no content. Check the logs above for warnings.")

    # --- Display Results (persists across reruns) ---
    if st.session_state.get("analysis_complete", False):
        report = st.session_state.analysis_report
        st.divider()
        st.write("### Analysis Complete")
        st.download_button(
            label="Download Analysis Report",
            data=report,
            file_name="IPA_Analysis_Report.md",
            mime="text/markdown"
        )
        with st.expander("Report Preview", expanded=False):
            st.markdown(report)

        if st.button("Clear Results", key="clear_results"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_report = ""
            st.rerun()


if __name__ == "__main__":
    main()
