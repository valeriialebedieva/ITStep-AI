import os
import warnings
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, initialize_agent, Tool

warnings.filterwarnings("ignore")

os.environ["GROQ_API_KEY"] = ""
os.environ["SERPER_API_KEY"] = ""

search = GoogleSerperAPIWrapper(type="places")


def restaurant_search(query):
    """
    Searches for restaurants and returns name, website, and rating.
    """
    results = search.results(query)
    places = results.get("places", [])

    if not places:
        return "No restaurants found for this area."

    formatted_results = ""
    for place in places[:3]:
        name = place.get("title", "N/A")
        rating = place.get("rating", "No rating")
        website = place.get("website", "No website available")

        formatted_results += f"- Name: {name}\n  Rating: {rating}\n  Website: {website}\n\n"

    return formatted_results


# 4. Wrap into a LangChain Tool
tools = [
    Tool(
        name="Restaurant_Recommendation",
        func=restaurant_search,
        description="Useful for finding restaurants in a specific city or area. Input should be a search query like 'best Italian restaurants in Kyiv'."
    )
]

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def main():
    print("(Ask for a restaurant or press Enter to exit)\n")

    while True:
        user_input = input("Human: ")
        if not user_input.strip():
            print("AI: Enjoy your meal! Goodbye.")
            break

        try:
            response = agent.run(user_input)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()