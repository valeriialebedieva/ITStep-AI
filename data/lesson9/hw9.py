import os
import warnings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage

warnings.filterwarnings("ignore")

os.environ["GROQ_API_KEY"] = ""


def chatbot():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    file_path = "return_policy.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            policy_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    instruction = f"You are a helpful assistant. Use this policy: {policy_content}"
    chat_history = []

    print("(Press Enter on an empty line to exit)\n")

    while True:
        user_input = input("Human: ")

        if not user_input.strip():
            print("AI: Goodbye!")
            break

        prompt = f"Instruction: {instruction}\n"
        for msg in chat_history:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            prompt += f"{role}: {msg.content}\n"
        prompt += f"Human: {user_input}\nAI:"

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            print(f"AI: {response.content}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.content))
        except Exception as e:
            print(f"AI Error: {e}")


if __name__ == "__main__":
    chatbot()