import os
import warnings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

os.environ["GROQ_API_KEY"] = ""

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)

# --- CHAIN 1: Goal -> Exercises ---
first_prompt = ChatPromptTemplate.from_template(
    "You are a fitness expert. The user's goal is: {goal}. "
    "Provide a simple list of 5-7 effective exercises for this goal."
)
chain_one = first_prompt | llm | StrOutputParser()

# --- CHAIN 2: Exercises + Context -> Final Plan ---
second_prompt = ChatPromptTemplate.from_template(
    "Based on these exercises: {exercises}. "
    "Create a training plan for a user with a {level} fitness level "
    "who has {time} hours per week to train. "
    "Format it as a clear weekly schedule."
)
chain_two = second_prompt | llm | StrOutputParser()

def generate_plan():
    user_goal = "muscle gain"
    user_level = "beginner"
    user_time = "3"


    # Run the first chain to get exercises
    exercises_list = chain_one.invoke({"goal": user_goal})
    print(f"\n[AI selected exercises]:\n{exercises_list}")

    # Run the second chain using the output from the first one
    final_plan = chain_two.invoke({
        "exercises": exercises_list,
        "level": user_level,
        "time": user_time
    })

    print("\n[Final Workout Plan]:")
    print(final_plan)

if __name__ == "__main__":
    generate_plan()