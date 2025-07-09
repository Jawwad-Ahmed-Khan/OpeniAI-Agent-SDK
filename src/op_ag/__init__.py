import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

def main():
    # Load the .env file for environment variables
    load_dotenv()
    deepseek_api_key = os.getenv("DEEP_SEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEP_SEEK_API_KEY environment variable is not set.")
    external_client = AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://your-site-url.com",  # Optional, helps OpenRouter track your usage
            "X-Title": "Deepseek Agent SDK App",           # Optional, app title
        }  )
    # Model wrapper
    model = OpenAIChatCompletionsModel(
        model="deepseek/deepseek-chat-v3-0324:free",
        openai_client=external_client   )
    # Runner config
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    # Define the agent
    agent = Agent(
        name="DeepseekAssistant",
        instructions="You are an intelligent and concise assistant powered by Deepseek.",
        model=model
    )

    # Run the agent
    result = Runner.run_sync(agent, "What is capital of France.", run_config=config)

    # Output results
    print("---------------| Final Output | -------------------")
    print(result.final_output)  

if __name__ == "__main__":
    main()
