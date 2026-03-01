import os
import warnings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


warnings.filterwarnings("ignore")

os.environ["GROQ_API_KEY"] = ""

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)


def summarize_history(messages):

    system_msg = [msg for msg in messages if isinstance(msg, SystemMessage)]
    chat_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    summary_prompt = (
        "Summarize the following conversation in a few sentences, "
        "keeping as many specific details as possible:\n\n"
    )
    for msg in chat_msgs:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        summary_prompt += f"{role}: {msg.content}\n"


    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    return system_msg + [AIMessage(content=f"Summary of previous chat: {summary_response.content}")]


def chat_bot():
    history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

    print("--- Detailed Summary Chatbot Active ---")
    print("(History will be summarized after 4 messages. Enter empty line to exit)\n")

    while True:
        user_input = input("Human: ")
        if not user_input.strip():
            break

        history.append(HumanMessage(content=user_input))
        chat_messages_count = len([msg for msg in history if not isinstance(msg, SystemMessage)])

        if chat_messages_count > 4:
            print("\n[System: Summarizing history to save memory...]")
            history = summarize_history(history)

        response = llm.invoke(history)
        print(f"AI: {response.content}\n")

        history.append(AIMessage(content=response.content))


if __name__ == "__main__":
    chat_bot()