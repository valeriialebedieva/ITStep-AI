import os
import warnings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage

# Silencing the annoying warnings
warnings.filterwarnings("ignore")

# 1. Paste your GROQ key here
os.environ["GROQ_API_KEY"] = "gsk_"


def chatbot():
    # Initialize the model via Groq (Free Tier)
    # This model is very smart and follows policy rules perfectly.
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    # Load the return policy document
    file_path = "return_policy.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            policy_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # Teacher's required instruction and history format
    instruction = f"You are a helpful assistant. Use this policy: {policy_content}"
    chat_history = []

    print("\n--- FREE STUDENT CHATBOT (NO BILLING) ---")
    print("(Press Enter on an empty line to exit)\n")

    while True:
        user_input = input("Human: ")

        if not user_input.strip():
            print("AI: Goodbye!")
            break

        # Build the specific prompt format requested in your assignment
        prompt = f"Instruction: {instruction}\n"
        for msg in chat_history:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            prompt += f"{role}: {msg.content}\n"
        prompt += f"Human: {user_input}\nAI:"

        try:
            # Get the answer
            response = llm.invoke([HumanMessage(content=prompt)])
            print(f"AI: {response.content}")

            # Save history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.content))
        except Exception as e:
            print(f"AI Error: {e}")


if __name__ == "__main__":
    chatbot()