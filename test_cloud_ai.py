from cloud_ai import ask_ai

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break
    print("AI:", ask_ai(msg))

