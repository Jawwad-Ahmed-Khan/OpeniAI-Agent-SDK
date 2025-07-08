import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

def main():

    # Load environment variables from .env file
    load_dotenv()
    gemini_api_key = os.getenv("GEMENI_API_KEY")

    if not gemini_api_key:
        raise ValueError("GEMENI_API_KEY environment variable is not set.")

    # Create an asynchronous OpenAI-style client for calling external APIs (e.g., Gemini)
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Define the model wrapper that communicates with the LLM using OpenAI-compatible schema
    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    # Runtime execution configuration (used to attach model, disable tracing, etc.)
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Define the agent, assign name, behavior (instructions), and model to use
    agent = Agent(name="Assitance", instructions="You are a helpful assistant.", model=model)

    # Run the agent synchronously using Runner with the given input and config
    result = Runner.run_sync(agent, "What are we disscussed now ", run_config=config)

    # Output full result from the agent
    print("---------------| Result | -------------------")
    print(result)
    print("---------------| Steps | -------------------")
    print("---------------| Final Output | -------------------")
    print(result.final_output)
    print(result.input)

if __name__ == "__main__":
    main()
