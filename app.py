import os
from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import speech, texttospeech, language_v1
import logging
import uuid

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\shriy\\Transcriptionapp\\angelic-triumph-436420-d5-bb4acdfe2f95.json"

# Define output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        logger.error("No audio file uploaded")
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    audio_content = audio_file.read()

    if not audio_content:
        logger.error("Empty audio content")
        return jsonify({'error': 'Empty audio content'}), 400

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    try:
        logger.info(f"Sending request to Google Speech-to-Text API. Audio content length: {len(audio_content)} bytes")
        response = client.recognize(config=config, audio=audio)
        logger.info(f"Received response from Google Speech-to-Text API: {response}")
        
        if not response.results:
            logger.warning("No transcription results returned from the API")
            return jsonify({'transcript': "No transcription available"}), 200

        transcript = response.results[0].alternatives[0].transcript
        logger.info(f"Transcription: {transcript}")

        # Perform sentiment analysis
        sentiment = analyze_sentiment(transcript)

        # Save transcription and sentiment to file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcript: {transcript}\n\nSentiment: {sentiment}")

        file_url = f'/output/{file_id}.txt'
        
        return jsonify({
            'transcript': transcript,
            'sentiment': sentiment,
            'file_url': file_url
        })
    
    except Exception as e:
        logger.error(f"Error in speech recognition: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred during transcription: {str(e)}'}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    text = request.json.get('text')
    if not text:
        logger.error("No text provided for synthesis")
        return jsonify({'error': 'No text provided'}), 400

    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        logger.info(f"Sending request to Google Text-to-Speech API. Text: {text}")
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.info("Received response from Google Text-to-Speech API")

        # Save synthesized speech to file
        file_id = str(uuid.uuid4())
        audio_file_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")
        text_file_path = os.path.join(OUTPUT_DIR, f"{file_id}.txt")

        with open(audio_file_path, "wb") as out:
            out.write(response.audio_content)

        # Perform sentiment analysis on the synthesized text
        sentiment = analyze_sentiment(text)

        with open(text_file_path, "w", encoding='utf-8') as out:
            out.write(f"Text: {text}\n\nSentiment: {sentiment}")

        audio_url = f'/output/{file_id}.mp3'
        text_url = f'/output/{file_id}.txt'

        return jsonify({
            'audio_url': audio_url,
            'text_url': text_url,
            'sentiment': sentiment
        })

    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during speech synthesis'}), 500

@app.route('/output/<filename>')
def serve_file(filename):
    try:
        return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': 'File not found'}), 404

def analyze_sentiment(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    try:
        response = client.analyze_sentiment(request={'document': document})
        sentiment = response.document_sentiment

        if sentiment.score > 0.25:
            return "Positive"
        elif sentiment.score < -0.25:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        return "Error in sentiment analysis"

if __name__ == '__main__':
    app.run(debug=True)