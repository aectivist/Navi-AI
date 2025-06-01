import whisper
import pyaudio
import wave
import numpy as np
import time

def Input():
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

    SILENCE_DURATION = 2
    SILENCE_THRESHOLD = 500
    silent_chunks = 0
    required_silent_chunks = int (SILENCE_DURATION * RATE / CHUNK)

    while True: 
        data = stream.read(1024)
        audio_data = np.frombuffer(data, dtype = np.int16)
        volume = np.sqrt(np.mean(audio_data.astype(np.float32)**2)) #RMS
        

        if volume < SILENCE_THRESHOLD:
            silent_chunks += 1
            seconds_silent = silent_chunks * CHUNK / RATE
            print(f"Silent for {seconds_silent:.2f} seconds")
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

    #WHISPER
    try:
        model = whisper.load_model("base")
        result = model.transcribe(r"C:\Users\aecti\OneDrive\Desktop\Projects\Elipson-AI\planning\recordsample.wav")
        print(result["text"])
    except Exception as e:
        print(e) 

def Navi():
    print("Hello world!")

Input()