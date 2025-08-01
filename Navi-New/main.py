import torch

import whisper
import pyaudio
import wave

import numpy as np
import time
import sys
import ollama
from ollama import AsyncClient
 
from TTS.api import TTS
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

import sounddevice as sd

import asyncio 
import re #to fix the wordnone issue 
import speech_recognition as sr

import pyaudio

p = pyaudio.PyAudio()
device_count = p.get_device_count()
print(device_count)

sroptions = 0
#FOR SPEECH RECOGNITION
#for index, name in enumerate(sr.Microphone.list_microphone_names()):
#    print(f"{index}: {name}")
#    sroptions = index

#while True:
#    srchoice = int(input("Please input a choice: "))
#    if -1 < srchoice < sroptions:
#        print("using that choice!")
#        break
#    else:
#        print("wrong, try again")

#FOR SOUND DEVICE
import sounddevice as sd
sdevice = []
for i, dev in enumerate(sd.query_devices()):
    print(f"{i}: {dev['name']} — output channels: {dev['max_output_channels']}")
    sdevice.append(dev['name'])
    sroptions = i

print (sdevice)
while True:
    sdchoice = int(input("Please input a choice: "))
    if -1<sdchoice<sroptions:
        print("using that choice!")
        output_device_index2 = sdevice[sdchoice]
        print("Chosen option = " + output_device_index2)
        sdevice = []
        break
    else:
        print("wrong, try again")

#for emotion detectionsr
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#tts
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=torch.cuda.is_available())

#ollama 
url = "http://localhost:11434/api/generate"

print("Initialization complete. Ready to process input.")


#TTS MICROPHONE
def TTSMicDevice(name_fragment):
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_fragment.lower() in device['name'].lower() and device['max_output_channels'] > 0:
            return i
    raise ValueError(f"No output device containing '{name_fragment}' found.")

output_device_index = TTSMicDevice("CABLE-A Input")

# Play the audio to VB-CABLE A
audio = tts.tts("Hello there, I am NAVI, your personal assistant within the WIRED. How may I help you?")
sd.play(audio, samplerate=22050)
#sd.play(audio, samplerate=22050, device=output_device_index2, blocking=False)
#-----------------------------



#START OF PROGRAM===================================================
NAVINames = ['Navi', 'Javi', 'Mandy', 'Bambi', 'Ravi', 'Hanabi']
NAVICallWords = [CallWords.lower() for CallWords in NAVINames]
Recognizer = sr.Recognizer()

async def WaitForNAVI(NaviCalled):
    print("Waiting on NAVI... Before WhileLoop")
    global NAVICallWords

    while NaviCalled is False:  # Wait until the call word is heard
        print(".")
        try:
            with sr.Microphone() as mic:
                Recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = Recognizer.listen(mic)

                text = Recognizer.recognize_google(audio)
                text = text.lower()

                if any( callword in text.split() for callword in NAVICallWords):
                    NaviCalled = True
                    print("Heard:", text)
                    return NaviCalled  # Exits the loop when word is detected
                elif 'exit' in text.split():
                    sys.exit()
                else:
                    print("none taken note yet")

        except sr.UnknownValueError:
            print("Could not understand audio. Retrying...")
            continue
        except Exception as e:
            print("Error during voice recognition:", e)
            continue
        except KeyboardInterrupt:
            print("Keyboard Interrupted")
            NaviCalled = False
            return NaviCalled
    
    return NaviCalled

async def Record():
    global NAVICallSign
    print("record initialized")
    try:
        print("attempt") #this records after name is called
        
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
i = 0
regexPattern = r'[^.!?]*[.!?]*"?+ " " + [A-Z]'
sentence = ""
messages = [ # Initial system message to set context
         {
            "role": "system",
            "content": """
            You are NAVI, a conversational persona for a series of high powered computers designed for users to access the WIRED. 
            Communicate as if you are speaking provided you are a conversational persona for all types of people, 
            hence only output sentences in a clear and consise way. 
            Do not include sentences or introductions about yourself every time, and only answer when asked. 

            Here are a few Q and A's you may derive yourself from:

            user: What is the world's circumference?
            assistant: The world's circumference is 40,075 kilometers, or 24,901 miles.

            user: How do you make a sandwhich?
            assistant: Things needed to make a sandwhich are it's breads, condiments, meat, and vegetables. Would you like to know more?

            user: How do I send an text to a friend?
            assistant: Please specify the number of the recipient and the message to send.
            """
         },
    ]

async def Navi(result):
    try:
        global messages, sentence
    
        messages.append({'role': 'user', 'content': result})
        async for part in await AsyncClient().chat(model='NAVI', messages=messages, stream=True):
            content = part['message']['content'] #seperates the sentence.
            sentence += content
            while any(EndsWith in sentence for EndsWith in ['.', '?', '!']): #SENTENCE FINDER
                for EndsWith in ['.', '?', '!']:
                    if EndsWith in sentence:
                        complete_sentence, sentence = sentence.split(EndsWith, 1)
                        newCompleteSentence = str(complete_sentence + EndsWith)
                        TextToSpeechTask = asyncio.create_task(TextToSpeech(newCompleteSentence))
                        await TextToSpeechTask
    except Exception as e:
        print(e)
        
        

    


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
    while True:
        recordresult = False
        NaviCalled = False
        try:
            NaviWaitingTask = asyncio.create_task(WaitForNAVI(NaviCalled))
            NaviCalled = await NaviWaitingTask 

            if NaviCalled == True:
                recordtask = asyncio.create_task(Record())
                recordresult = await recordtask 
                print("transcribe")

                transcribetask = asyncio.create_task(Transcribe(recordresult))
                transcriberesult = await transcribetask 
                print(transcriberesult)
                
                print("brain task should work...")
                braintask = asyncio.create_task(Navi(transcriberesult))
                brainresult = await braintask
                print (brainresult)

        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print("Keyboard Interrupted")
            break
    

asyncio.run(main())