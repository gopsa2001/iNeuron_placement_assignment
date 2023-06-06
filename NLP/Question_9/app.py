import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os

# Function to transcribe an audio file to text
def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the entire audio file
    text = r.recognize_google(audio)
    return text

# Function to translate text to a different language
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Function to convert text to speech and save as an audio file
def text_to_speech(text, output_file, target_language):
    tts = gTTS(text, lang=target_language)
    tts.save(output_file)

# Example usage
audio_file = "harvard.wav"
output_text_file = "output.txt"
translated_output_file = "translated_output.mp3"
target_language = "fr"  # Target language (e.g., "fr" for French)

# Transcribe audio file to text
transcribed_text = transcribe_audio(audio_file)

# Save the transcribed text to a file
with open(output_text_file, "w") as file:
    file.write(transcribed_text)

# Translate the text to the target language
translated_text = translate_text(transcribed_text, target_language)

# Convert the translated text to speech and save as an audio file
text_to_speech(translated_text, translated_output_file, target_language)

# Print the translated text and the paths of the output files
print("Translated Text:", translated_text)
print("Transcribed Text File:", os.path.abspath(output_text_file))
print("Translated Audio File:", os.path.abspath(translated_output_file))
