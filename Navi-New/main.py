import torch

import whisper
import pyaudio
import wave

import numpy as np
import time
import ollama
from ollama import AsyncClient
 
from TTS.api import TTS
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

import sounddevice as sd

import asyncio 

#for emotion detection
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#tts
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=torch.cuda.is_available())

#ollama 
url = "http://localhost:11434/api/generate"

print("Initialization complete. Ready to process input.")

messages=[]

shareR = 0
lock = asyncio.Lock()

async def Record():
   try:
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

        sound_file = wave.open(r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-AI\main\input\record.wav", "wb")
        sound_file.setnchannels(CHANNELS)
        sound_file.setsampwidth(audio.get_sample_size(FORMAT))
        sound_file.setframerate(RATE)
        sound_file.writeframes(b''.join(frames))
        sound_file.close()

        return True
   except Exception as e:
       print(e)
       return False


async def Transcribe(audioFlag):
        #WHISPER
    if audioFlag:
        try:
            model = whisper.load_model("base")
            result = model.transcribe(r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-AI\main\input\record.wav")
            NAVI_Input = str(result["text"])
            return NAVI_Input

        except Exception as e:
            print(e) 


#NAVI
async def Navi(result):
    global messages
    i = 0
    message = {'role': 'user', 'content': result}
    async for part in await AsyncClient().chat(model='NAVI', messages=[message], stream=True):
        result = print(part['message']['content'], end='', flush=True)
        TextToSpeechTask = asyncio.create_task(TextToSpeech(str(result)))
        await TextToSpeechTask
        
        i =+ 1
        print(i)
    


async def Emotion(response):
        emotion_labels = emotion(response)
        print(emotion_labels)

async def TextToSpeech(response):
    #text to speech
    print(response)
    audio_output = tts.tts(response)
    sd.play(audio_output, samplerate=22050)  
    sd.wait()  
    

async def main():
    #audio
    recordresult = False
    recordtask = asyncio.create_task(Record())
    recordresult = await recordtask

    transcribetask = asyncio.create_task(Transcribe(recordresult))
    transcriberesult = await transcribetask 
    print(transcriberesult)
        
    #brain
    braintask = asyncio.create_task(Navi(transcriberesult))
    brainresult = await braintask
    print (brainresult)
    

asyncio.run(main())