base_path = r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\planning\dataset_folder\wavs"
paths = [f"r'{base_path}\\{i}.wav'" for i in range(1, 225)]
output_line = ", ".join(paths)
print(output_line)
