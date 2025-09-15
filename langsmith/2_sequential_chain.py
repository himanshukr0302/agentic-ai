import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the traceable decorator from the langsmith package
from langsmith import traceable

# Import LangChain components
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# The @traceable decorator tells LangSmith to track this function
# We explicitly set the project_name here, which is the most reliable method.
@traceable(run_type="chain", name="Sequential Chain", project_name="Sequential LLM App")
def run_report_chain(topic: str):
    """Generates a report on a topic and then summarizes it."""

    prompt1 = PromptTemplate(
        template='Generate a detailed report on {topic}',
        input_variables=['topic']
    )
    prompt2 = PromptTemplate(
        template='Generate a 5 pointer summary from the following text \n {text}',
        input_variables=['text']
    )

    model1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
    model2 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)

    parser = StrOutputParser()

    chain = prompt1 | model1 | parser | prompt2 | model2 | parser

    # Define the config for the LangChain components within the trace
    config = {
        'tags': ['llm app', 'report generation', 'summarization'],
        'metadata': {'model1': 'gemini-1.5-flash', 'model1_temp': 0.7}
    }

    print("Running the chain...")
    result = chain.invoke({'topic': topic}, config=config) #type:ignore
    print("Chain finished.")
    
    return result

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # Call the decorated function to run your logic
    final_result = run_report_chain("Rise of AI in Gaming")
    print("\n--- Final Summary ---")
    print(final_result)