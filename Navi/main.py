import torch

import whisper
import pyaudio
import wave

import numpy as np
import time
import sys
import ollama
import subprocess
from ollama import AsyncClient
 
from TTS.api import TTS
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

import sounddevice as sd

import asyncio 
import re #to fix the wordnone issue 
import speech_recognition as sr

#-0-----------Literal fucking mess
import tkinter as tk
import vlc
root = tk.Tk()

root.title("VLC mood player")

vlc_instance=vlc.Instance()
player = vlc_instance.media_player_new()

def MoodPlayer(mood): 
    global player
    if mood == 'happy':
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\happy2.mp4"
    elif mood == 'neutral':
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\happy.mp4"
    elif mood == 'serious':
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\serious2.mp4"
    elif mood == 'angry':
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\angry2.mp4"
    elif mood == 'sad':
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\sad.mp4"
    else:
        file_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\AI\NAVI-OFFICIAL\Navi-AI\Navi\assets\happy2.mp4"
    
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return
    
    file = vlc.MediaPlayer(file_path)
    player.play()
    loop()

def loop():
    if player.get_state() == vlc.State.Ended:
        player.stop()
        player.play()
    root.after(500, loop)

MoodPlayer("happy")

#---Fix this later, I'm too fucking tired for this bullshit

import pyaudio
from task import NAVI_FUNCTION

p = pyaudio.PyAudio()
device_count = p.get_device_count()
print(device_count)

#for emotion detectionsr
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#tts
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=torch.cuda.is_available())

#ollama 
url = "http://localhost:11434/api/generate"


#VSEEFACE
print("Initialization complete. Ready to process input.")

VseeFaceInit = subprocess.Popen(r"C:\Users\aecti\Downloads\vtuber\VSeeFace-v1.13.38c2\VSeeFace\VSeeFace.exe")
root = tk.Tk()
#Checking to see if the user is done configuring VSEEFACE 
while True:
    ReadyOption = input("Are you ready? Y/N: ")
    if  ReadyOption == 'y' or ReadyOption == 'Y':
        print("Lets start!")
        break
    elif  ReadyOption == 'n' or ReadyOption == 'N':
        print("good bye!")
        VseeFaceInit.terminate()
        sys.exit(0)
        break
    else:
        print("Try again")
    
#TTS MICROPHONE
def TTSMicDevice(name_fragment):
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_fragment.lower() in device['name'].lower() and device['max_output_channels'] > 0:
            return i
    raise ValueError(f"No output device containing '{name_fragment}' found.")



# Play TTS for test


import threading
def TTSMicDevice(name_fragment):
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_fragment.lower() in device['name'].lower() and device['max_output_channels'] > 0:
            return i
    raise ValueError(f"No output device containing '{name_fragment}' found.")

def match_channels(audio_data, channels):
    return np.tile(audio_data.reshape(-1, 1), (1, channels))

def play_on_device(audio_data, device, samplerate, channels):
    with sd.OutputStream(device=device, samplerate=samplerate, channels=channels, dtype='float32') as stream:
        stream.write(audio_data)

async def choiceForTTS(audio):
    output_device_index = TTSMicDevice("CABLE-A")
    output_speaker = TTSMicDevice("Speakers")

    channels = 2
    samplerate = 22050

    audio_data = np.array(audio, dtype=np.float32)
    audio_multich = match_channels(audio_data, channels)

    # Create two threads to play audio aatst
    thread1 = threading.Thread(target=play_on_device, args=(audio_multich, output_speaker, samplerate, channels))
    thread2 = threading.Thread(target=play_on_device, args=(audio_multich, output_device_index, samplerate, channels))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
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
    
        messages = NAVI_FUNCTION(result, messages)
        async for part in await AsyncClient().chat(model='NAVI', messages=messages, stream=True):
            content = part['message']['content'] #seperates the sentence.
            sentence += content
            while any(EndsWith in sentence for EndsWith in ['.', '?', '!']): #SENTENCE FINDER
                for EndsWith in ['.', '?', '!']:
                    if EndsWith in sentence:
                        complete_sentence, sentence = sentence.split(EndsWith, 1)
                        newCompleteSentence = str(complete_sentence + EndsWith)
                        
                        audio_output = tts.tts(newCompleteSentence)
                        EmotionTask = asyncio.create_task(Emotion(newCompleteSentence))
                        TextToSpeechDouble = asyncio.create_task(choiceForTTS(audio_output))
                        await TextToSpeechDouble
                        await EmotionTask
    except Exception as e:
        print(e)
        
        

    


async def Emotion(response):
        emotion_labels = emotion(response)
        MOOD = emotion_labels[0]['label']
        MoodPlayer(MOOD)
        print(emotion_labels)




async def main():
    
    audio = tts.tts("Hello there, I am NAVI, your personal assistant within the WIRED. How may I help you?")
    TTSVoiceCheck = asyncio.create_task(choiceForTTS(audio))
    await TTSVoiceCheck
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
    VseeFaceInit.terminate()

asyncio.run(main())
root.mainloop()