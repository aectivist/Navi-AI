import whisper
import pyaudio
import wave

#RECORDER
audio = pyaudio.PyAudio() #Implements pyaudio

#STREAM: 
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=10)

#Frames (because recorded through individual frames like animation but for voice):
frames = []

try: #starts to loop/record UNTIL keyboard interrupts
    while True:
        data = stream.read(1024)
        frames.append(data)
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
audio.terminate()

sound_file = wave.open(r"C:\Users\aecti\OneDrive\Desktop\Projects\Elipson-AI\planning\recordsample.wav", "wb")
sound_file.setnchannels(1)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(44100)
sound_file.writeframes(b''.join(frames))
sound_file.close()

#WHISPER
try:
    model = whisper.load_model("base")
    result = model.transcribe(r"C:\Users\aecti\OneDrive\Desktop\Projects\Elipson-AI\planning\recordsample.wav")
    print(result["text"])
except Exception as e:
    print(e)



