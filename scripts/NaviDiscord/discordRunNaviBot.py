import subprocess

node_process = subprocess.Popen(["node", "scripts/NaviDiscord/recordVC/index.js"])
py_process = subprocess.Popen([r"C:\Users\aecti\OneDrive\Desktop\Projects\NAVI-AI\.venv\Scripts\python.exe", "scripts/NaviDiscord/discordMain.py"])

node_process.wait()
py_process.wait()