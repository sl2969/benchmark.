import os
import openai
import speech_recognition as sr
import tempfile
import logging

# --- API SETUP ---
os.environ["OPENAI_API_KEY"] = "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"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI()  # OpenAI Client

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- AUDIO TRANSCRIPTION FUNCTION ---
def transcribe_audio(audio_data: sr.AudioData) -> str:
    """
    Transcribes audio using OpenAI Whisper and returns text.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data.get_wav_data())
        tmp_filename = tmp_file.name

    try:
        with open(tmp_filename, "rb") as audio_file:
            logging.info("Sending audio to OpenAI Whisper API...")
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcript_response.text.strip()
        if not text:
            raise ValueError("No transcription returned from the API.")
        return text

    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return ""

    finally:
        os.remove(tmp_filename)  # Cleanup temporary file

# --- MAIN FUNCTION ---
def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    logging.info("Calibrating microphone... Please wait.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    logging.info("Calibration complete. Start speaking... (Press Ctrl+C to finish)")

    transcript_lines = []

    try:
        while True:
            with mic as source:
                logging.info("Listening...")
                audio_data = recognizer.listen(source)

            try:
                logging.info("Transcribing...")
                text = transcribe_audio(audio_data)
                if text:
                    logging.info(f"Transcribed: {text}")
                    transcript_lines.append(text)
                else:
                    logging.warning("No text transcribed.")
            except Exception as e:
                logging.error(f"Failed to transcribe audio: {e}")

    except KeyboardInterrupt:
        logging.info("Stopping transcription.")
        full_transcript = "\n".join(transcript_lines)

        # Save transcript to a file
        with open("transcript.txt", "w") as file:
            file.write(full_transcript)

        logging.info("Transcript saved to transcript.txt")
        print("\n--- Complete Transcript ---\n")
        print(full_transcript)
        print("\n---------------------------\n")

if __name__ == "__main__":
    main()
