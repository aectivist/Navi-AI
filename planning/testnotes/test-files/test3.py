import ollama 

response = ollama.list()


#chat example WITH STREAM
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role":"user", "content": "Hello! Who are you?"}
    ],
    stream=True,
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)

