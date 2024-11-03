import streamlit as st
import openai
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client with the API key
client = OpenAI(
    api_key=st.secrets["openai_api_key"]  # Ensure this key exists in Streamlit secrets
)

def call_chatgpt(prompt, model="gpt-4", max_tokens=1500, temperature=0.3, retries=3):
    """
    Sends a prompt to the OpenAI ChatGPT API and returns the response.
    Includes basic error handling and rate limiting.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert qualitative researcher specializing in Interpretative Phenomenological Analysis (IPA).",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Log the response for debugging
        logger.info(f"API Response: {response}")
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError:
        if retries > 0:
            st.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            return call_chatgpt(prompt, model, max_tokens, temperature, retries - 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            logger.error("Rate limit exceeded.")
            return ""
    except openai.error.OpenAIError as e:
        st.error(f"An OpenAI error occurred: {e}")
        logger.error(f"OpenAIError: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
        return ""

def convert_to_markdown(data):
    """Converts the analysis data to Markdown format."""
    markdown = ""
    pets = data.get("personal_experiential_themes", [])
    if not pets:
        st.error("No Personal Experiential Themes found in the data.")
        logger.error("No Personal Experiential Themes found in the data.")
        return ""

    for pet in pets:
        pet_title = pet.get('personal_experiential_theme', 'Untitled Theme')
        markdown += f"## {pet_title}\n\n"
        description = pet.get('description', 'N/A')
        markdown += f"**Description:** {description}\n\n"

        extracts = pet.get("extracts", [])
        markdown += f"**Extracts:**\n"
        if extracts:
            for extract in extracts:
                markdown += f"- {extract}\n"
        else:
            markdown += "- No extracts provided.\n"

        analytic_comments = pet.get("analytic_comments", [])
        markdown += "\n**Analytic Comments:**\n"
        if analytic_comments:
            for comment in analytic_comments:
                markdown += f"- {comment}\n"
        else:
            markdown += "- No analytic comments provided.\n"

        markdown += "\n"
    return markdown

def convert_to_text(data):
    """Converts the analysis data to plain text format."""
    text = ""
    pets = data.get("personal_experiential_themes", [])
    if not pets:
        st.error("No Personal Experiential Themes found in the data.")
        logger.error("No Personal Experiential Themes found in the data.")
        return ""

    for pet in pets:
        pet_title = pet.get('personal_experiential_theme', 'Untitled Theme')
        text += f"Personal Experiential Theme (PET): {pet_title}\n"
        description = pet.get('description', 'N/A')
        text += f"  Description: {description}\n"
        extracts = pet.get("extracts", [])
        text += f"  Extracts:\n"
        if extracts:
            for extract in extracts:
                text += f"    - {extract}\n"
        else:
            text += "    - No extracts provided.\n"
        analytic_comments = pet.get("analytic_comments", [])
        text += f"  Analytic Comments:\n"
        if analytic_comments:
            for comment in analytic_comments:
                text += f"    - {comment}\n"
        else:
            text += "    - No analytic comments provided.\n"
        text += "\n"
    return text

def stage4_write_up_pet(pets, transcript):
    """Stage 4: Writing up Personal Experiential Themes (PETs) with extracts and analytic comments."""
    prompt = f"""
Using the following Personal Experiential Themes (PETs) from an IPA analysis, concisely write up each theme.
For each PET, include a brief description, relevant extracts from the transcript, and analytic comments.
Limit the number of extracts and comments to a maximum of 2 per theme to keep the response concise.

Personal Experiential Themes (PETs):
{json.dumps(pets, indent=2)}

Transcript:
{transcript}

Provide the output in a well-formatted JSON structure as follows:
{{
  "personal_experiential_themes": [
    {{
      "personal_experiential_theme": "Theme Title 1",
      "description": "Brief description of Theme Title 1",
      "extracts": [
        "Relevant extract from the transcript",
        "Another relevant extract"
      ],
      "analytic_comments": [
        "Analytic comment related to Theme Title 1",
        "Another analytic comment"
      ]
    }},
    ...
  ]
}}

Ensure the JSON is complete and properly formatted, and do not include any text outside the JSON structure.
"""
    return call_chatgpt(prompt, max_tokens=1500)

def ipa_analysis_pipeline(transcript, output_filename):
    """Runs the full IPA analysis pipeline on a given transcript."""
    try:
        transcript_text = transcript.read().decode("utf-8")
        if not transcript_text.strip():
            st.error("The uploaded transcript file is empty.")
            return
    except Exception as e:
        st.error(f"Error reading the transcript file: {e}")
        logger.error(f"Error reading the transcript file: {e}")
        return

    # Stage 1 to 3 code remains the same, ensure max_tokens are appropriately set.

    # Stage 4
    st.write("### Stage 4: Writing Up PETs with Extracts and Comments...")
    with st.spinner("Writing up PETs..."):
        write_up_json = stage4_write_up_pet(pets, transcript_text)

    if write_up_json:
        try:
            write_up = json.loads(write_up_json)
            st.success("Stage 4 completed successfully.")
        except json.JSONDecodeError as e:
            st.error("Error parsing JSON from Stage 4. Please check the API response.")
            logger.error(f"Error parsing JSON from Stage 4: {e}")
            st.write("API Response Content:")
            st.code(write_up_json)
            write_up = {}
    else:
        write_up = {}

    if write_up:
        st.write("### Final Analysis:")
        markdown_content = convert_to_markdown(write_up)
        if not markdown_content:
            st.error("Failed to generate Markdown content.")
            return
        st.markdown(markdown_content)

        # Provide download buttons
        st.write("### Download Results:")

        # Prepare file names
        base_filename = os.path.splitext(os.path.basename(output_filename))[0]
        markdown_filename = f"{base_filename}.md"
        text_filename = f"{base_filename}.txt"
        json_filename = f"{base_filename}.json"

        # Encode the contents
        markdown_bytes = markdown_content.encode("utf-8")
        text_content = convert_to_text(write_up)
        text_bytes = text_content.encode("utf-8")
        json_bytes = json.dumps(write_up, indent=2).encode("utf-8")

        # Create download buttons
        st.download_button(
            label="Download Markdown",
            data=markdown_bytes,
            file_name=markdown_filename,
            mime="text/markdown",
        )

        st.download_button(
            label="Download Text",
            data=text_bytes,
            file_name=text_filename,
            mime="text/plain",
        )

        # Optionally provide JSON download
        st.download_button(
            label="Download JSON",
            data=json_bytes,
            file_name=json_filename,
            mime="application/json",
        )
    else:
        st.error("Stage 4 failed. Analysis incomplete.")

def main():
    st.title("Interpretative Phenomenological Analysis (IPA) Tool")

    st.write(
        """
    Upload your interview transcript and specify the desired output file name to perform IPA using ChatGPT.
    """
    )

    uploaded_file = st.file_uploader("Choose a transcript text file", type=["txt"])
    output_filename = st.text_input("Enter the desired output file name without extension (e.g., output_analysis)")

    if st.button("Run IPA Analysis"):
        if uploaded_file and output_filename:
            # Ensure output_filename does not have an extension
            output_filename = os.path.splitext(output_filename)[0]
            ipa_analysis_pipeline(uploaded_file, output_filename)
        else:
            st.warning("Please upload a transcript file and specify an output file name.")

if __name__ == "__main__":
    main()
    
