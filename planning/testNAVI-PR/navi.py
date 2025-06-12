import torch

import whisper
import pyaudio
import wave

import numpy as np
import time
import ollama
 
from TTS.api import TTS
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

import sounddevice as sd

#for emotion detection
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#tts
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=torch.cuda.is_available())

#ollama 
url = "http://localhost:11434/api/generate"

print("Initialization complete. Ready to process input.")

class Navi:
    def Record():
        #RECORDER
        audio = pyaudio.PyAudio() #Implements pyaudio

        #STREAM: 
        RATE = 44100
        CHUNK=1024
        CHANNELS = 1
        FORMAT = pyaudio.paInt16
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        #Frames (because recorded through individual frames like animation but for voice):
        frames = []

        print("recording now:") 

        SILENCE_DURATION = 1.5
        SILENCE_THRESHOLD = 500
        silent_chunks = 0
        required_silent_chunks = int (SILENCE_DURATION * RATE / CHUNK)

        while True: 
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype = np.int16)
            volume = np.sqrt(np.mean(audio_data.astype(np.float32)**2)) #RMS

            if volume < SILENCE_THRESHOLD:
                silent_chunks += 1
                #seconds_silent = silent_chunks * CHUNK / RATE
                #print(f"Silent for {seconds_silent:.2f} seconds")
                frames.append(data)
            else:
                silent_chunks = 0
                frames.append(data)
            
            if silent_chunks > required_silent_chunks:
                print("silence detected, stopping:")
                break
                

        stream.stop_stream()
        stream.close()
        audio.terminate()

        sound_file = wave.open(r"C:\Users\aecti\OneDrive\Desktop\Projects\Elipson-AI\planning\recordsample.wav", "wb")
        sound_file.setnchannels(CHANNELS)
        sound_file.setsampwidth(audio.get_sample_size(FORMAT))
        sound_file.setframerate(RATE)
        sound_file.writeframes(b''.join(frames))
        sound_file.close()


    def Transcribe():
            #WHISPER
        try:
            model = whisper.load_model("base")
            result = model.transcribe(r"C:\Users\aecti\OneDrive\Desktop\Projects\Elipson-AI\planning\recordsample.wav")
            NAVI_Input = str(result["text"])
            return NAVI_Input

        except Exception as e:
            print(e) 

    #NAVI
    def Navi(result):
        #ollama response
        res = ollama.generate(model="NAVI", prompt=result)
        output = str(res["response"])
        print(output)
        return output


    def TextToSpeech(response):
        #emotion detection
            emotion_labels = emotion(response)
            print(emotion_labels)

            #text to speech
            audio_output = tts.tts(response)
            sd.play(audio_output, samplerate=22050)  
            sd.wait()  

while True:
    print("Input:")
    input = input()
    NaviOutput = Navi.Navi(input)
    Navi.TextToSpeech(NaviOutput)
    time.sleep(1)  # Wait for a second before the next input
    