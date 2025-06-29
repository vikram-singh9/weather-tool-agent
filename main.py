import os
from dotenv import load_dotenv
from agents import AsyncOpenAI,OpenAIChatCompletionsModel,Agent,Runner,set_tracing_disabled,function_tool
import asyncio
import chainlit as cl
import requests

load_dotenv()
set_tracing_disabled(True)





@function_tool
def get_weather(city: str) -> str:
    """
    Get real-time weather information for a given city using OpenWeatherMap API.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OpenWeatherMap API key is not set."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"].capitalize()
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (
            f"The current weather in {city} is {description} with a temperature of "
            f"{temperature}°C (feels like {feels_like}°C) and humidity at {humidity}%."
        )

    except requests.RequestException as e:
        return f"Failed to fetch weather data for {city}: {e}"

    except KeyError:
        return f"Sorry, could not find weather info for '{city}'."




@cl.on_chat_start
async def start():
    MODEL_NAME = "gemini-2.0-flash"
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    
    external_client = AsyncOpenAI(
        api_key= API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    model = OpenAIChatCompletionsModel(
         model=MODEL_NAME,
         openai_client=external_client,
    )
    
    cl.user_session.set("chat_history",[]) # Initialize chat history

    assistant = Agent(
        name = "weather assistant",
        instructions= f"You are a helpful assistant that provides real time weather information.",
        model=model,
        tools=[get_weather]
    )
    
    cl.user_session.set("agent", assistant) # Store the agent in user session
    
    await cl.Message(
        content="Welcome to the Weather Assistant! You can ask me about the weather in any city.",
    ).send()
    
@cl.on_message
async def main(message: cl.Message):
    msg = await cl.Message(content="Thinking...").send()
    assistant = cl.user_session.get("agent") # Retrieve the agent from user session
    history = cl.user_session.get("chat_history") or []
     
    history.append({"role": "user", "content": message.content}) 
    result = await Runner.run(
        starting_agent=assistant,
        input=history,
    )   
    msg.content = result.final_output
    await msg.update()
    cl.user_session.set("chat_history", result.to_input_list())
    print(result.final_output)
        


if __name__ == "__main__":
    asyncio.run(start())