import streamlit as st
import os
import asyncio
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Configure IBM Watson Speech to Text
def configure_speech_to_text():
    authenticator = IAMAuthenticator('MfytpqW1rNUO_Cbw6uJW_d3wpSiQJ9PDya5N5u0l_TH1')
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url('https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/25abacd4-0391-4270-957f-a33136701ebb')
    return speech_to_text
# Configure WatsonX model for text generation
def configure_watsonx():
    credentials = {
        "url": os.getenv("https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29", "https://us-south.ml.cloud.ibm.com"),
        "apikey": "5XJ0ZRG-BNqlnNeegvvwribw8zXSZuhzfF_D8ibdrL0I"
    }
    #"project_id": "988f1fe5-26dc-4d23-bde9-eaf6f6431cee"
    project_id = "c20503f0-cbd7-42e7-bb7d-4eb5487e3db4"
    # Adjust model parameters for optimal summarization
    model_id = "mistralai/mistral-large"
    parameters = {
        GenParams.DECODING_METHOD: "greedy",  # Greedy decoding for more deterministic results
        GenParams.MIN_NEW_TOKENS: 100,        # Ensure the summary is sufficiently detailed
        GenParams.MAX_NEW_TOKENS: 2000,        # Set a limit to avoid overly verbose outputs
        GenParams.TOP_P: 0.9,                 # Use nucleus sampling to maintain diversity in output
        GenParams.TEMPERATURE: 0.7            # Adjust temperature to control randomness; lower values yield more focused outputs
    }
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    return model
async def generate_summary_async(model, transcript):
    prompt = (
        "You are analyzing a transcript from a call center conversation in a telecommunications company. "
        "Your task is to provide a summary that highlights the main points "
    )
    full_prompt = f"{prompt}\n\nTranscript:\n{transcript}"
    response = await asyncio.to_thread(model.generate, full_prompt)
    return response.get("results")[0]["generated_text"]
# Streamlit app setup
def main():
    st.title("Transcription and Summary Generator")
    # Upload file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        st.info("Transcribing the audio...")
        # Initialize the Speech to Text service
        speech_to_text = configure_speech_to_text()
        # Transcribe the uploaded audio file
        try:
            result = speech_to_text.recognize(
                audio=uploaded_file,
                content_type=uploaded_file.type,
                model='en-US_BroadbandModel'
            ).get_result()
            # Extract the transcript text
            transcript = ' '.join([r['alternatives'][0]['transcript'] for r in result['results']])
            st.success("Transcription complete!")
            st.text_area("Transcript", value=transcript, height=150)
            st.info("Generating summary...")
            # Initialize the WatsonX model for text summarization
            model = configure_watsonx()
            # Generate the summary asynchronously
            call_summary = asyncio.run(generate_summary_async(model, transcript))
            st.success("Summary generated!")
            #/////
            st.text_area("Summary", value=call_summary, height=150)
        except Exception as e:
            st.error(f"Error during transcription or summarization: {str(e)}")
if __name__ == "__main__":
    main()





# \Users\SESONAMaraxaba\Desktop\Text_to_Speech\Speech_text.py

