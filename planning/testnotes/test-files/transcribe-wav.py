import os
import whisper

# Define the path where the .wav files are located
wav_directory = r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs"

# Output text file path
output_file = os.path.join(wav_directory, "list.txt")

# Define the range of .wav files (1 to 224 — end is exclusive)
wav_files_range = range(1, 225)  # Adjust this based on your file count

# Initialize the Whisper model
model = whisper.load_model("base")

# List to store results
file_and_transcripts = []

# Process each file
for i in wav_files_range:
    wav_file = os.path.join(wav_directory, f"{i}.wav")

    if os.path.exists(wav_file):
        try:
            result = model.transcribe(wav_file)
            transcript = result["text"].strip()
            print(f"Processed {wav_file}")
        except Exception as e:
            print(f"Error transcribing {wav_file}: {e}")
            continue

        file_and_transcripts.append(f"{i}|wav-{i}|{transcript}")
    else:
        print(f"File not found: {wav_file}")

# Write to the output file
with open(output_file, "w", encoding="utf-8") as f:
    for line in file_and_transcripts:
        f.write(line + "\n")

print(f"Transcripts written to '{output_file}' successfully.")
