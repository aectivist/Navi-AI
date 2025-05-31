import ollama 
url = "http://localhost:11434/api/generate"

#show
#print (ollama.show("llama3.2"))

ollama.create (
    model="knowitall",
    from_="llama3.2", 
    system="You are Elipson, a very smart assistant who answers questions succinctly and informatively, providing summarized information regarding necessary topics.",
    parameters={"temperature": 0.3}
    )

res = ollama.generate(model="knowitall", prompt="who are you?")
print(res["response"])

ollama.delete("knowitall")