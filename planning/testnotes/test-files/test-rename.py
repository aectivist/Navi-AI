import os 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def rename_wave_files(folder_path):
    files = os.listdir(folder_path)
    wav_files = [f for f in files if f.lower().endswith('.wav')]

    print(f'Folder path: {folder_path}')
    print(f'All files in the folder: {files}')
    print(f'Wave files found{len(wav_files)}')

    for index, wav_files in enumerate(wav_files, start=1):
        old_path = os.path.join(folder_path, wav_files)
        new_path = os.path.join(folder_path, f'{index}.wav')
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')

if __name__ == "__main__":
    folder_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\wav\Futaba-Sakura"
    rename_wave_files(folder_path)
    print("Wave files renamed successfully!")



