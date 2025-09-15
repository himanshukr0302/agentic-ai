import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

search_tool = DuckDuckGoSearchRun()

# --- MODIFIED FUNCTION ---
# Updated to use the correct URL and key format for weatherapi.com
@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city using weatherapi.com
  """
  # Load the API key securely from your environment
  api_key = os.getenv("WEATHERAPI_API_KEY") # <-- Using a clearer variable name
  if not api_key:
      return "WeatherAPI.com API key not found. Please set WEATHERAPI_API_KEY in your .env file."
  
  # The URL is now for weatherapi.com and uses 'key' and 'q' as parameters
  url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
  
  try:
    response = requests.get(url)
    response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
    return response.json()
  except requests.exceptions.RequestException as e:
    return f"Error fetching weather data: {e}"


# --- THE REST OF THE CODE REMAINS THE SAME ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

client = Client()
tracer = LangChainTracer(project_name="agent-app-demo", client=client)
config = {"callbacks": [tracer]}

question = "What is the current temperature of Greater Noida?"
response = agent_executor.invoke({"input": question}, config=config) #type:ignore

print("\n--- Final Answer ---")
print(response['output'])