import os
import warnings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema import HumanMessage

warnings.filterwarnings("ignore")

os.environ["GROQ_API_KEY"] = ""
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

topic = "Photoshop Basics"
audience = "Children aged 10-12"

# --- 1. Zero-shot Approach ---
zero_shot_template = """
Create a detailed lesson plan for a course on the topic: {topic}.
Target audience: {audience}.
The plan should include a list of modules and key learning objectives.
"""

zero_shot_prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template=zero_shot_template
)

# --- 2. Few-shot Approach ---
examples = [
    {
        "topic": "Python Programming",
        "audience": "beginners",
        "plan": "1. Intro to Python (Variables, Types) 2. Control Flow (If, Loops) 3. Functions."
    },
    {
        "topic": "Digital Marketing",
        "audience": "professionals",
        "plan": "1. Advanced Analytics 2. Strategy Optimization 3. ROI Calculation."
    }
]

example_formatter_template = """
Topic: {topic}
Audience: {audience}
Plan: {plan}
"""
example_prompt = PromptTemplate(
    input_variables=["topic", "audience", "plan"],
    template=example_formatter_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Create a course plan based on the following examples:",
    suffix="Topic: {topic}\nAudience: {audience}\nPlan:",
    input_variables=["topic", "audience"]
)

def run_task():

    print("--- Zero-shot Result ---")
    formatted_zero = zero_shot_prompt.format(topic=topic, audience=audience)
    print(llm.invoke([HumanMessage(content=formatted_zero)]).content)

    print("\n" + "=" * 50 + "\n")

    # Run Few-shot
    print("--- Few-shot Result ---")
    formatted_few = few_shot_prompt.format(topic=topic, audience=audience)
    print(llm.invoke([HumanMessage(content=formatted_few)]).content)


if __name__ == "__main__":
    run_task()