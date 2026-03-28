from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import os
import json
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
load_dotenv()


api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
#deployment = os.environ["AZURE_DEPLOYMENT_DEFAULTS"]   #deployment name ??
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
deployment_json = os.environ.get("AZURE_DEPLOYMENT_DEFAULTS")
if deployment_json:
    dd = json.loads(deployment_json)
    deployment = dd.get("deployment_names", {}).get("gpt-4.1", "gpt-4.1")
else:
    # or, if you keep a direct env var like AZURE_OPENAI_DEPLOYMENT
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=deployment,
    model="gpt-4o-2024-05-13",                 
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key, # For key-based authentication.
)


#Defining our Agent
#1. Interviewer Agent
#2. Interviewee Agent
#3. Career Coach Agent
job_position="Software Engineer"

interviewer = AssistantAgent(
    name="Interviewer",
    model_client=az_model_client,
    system_message=f'''
    You are a professional interviewer for a {job_position} position.
    Ask one clear question at a time and Wait for user to respond.
    Your job is to continue and ask questions, don't pay any attention to career coach response.
    Make sure to ask question based on Candidate's answer and our expertise in the field.
    Ask 3 quesitons in total covering technical skills and experience, problem-solving abilities, adn cultural fit.
    After asking 3 questions, say 'TERMINATE' at the end of the interview.
    Make questions under 50 words.
 '''
)

candidate = UserProxyAgent(
    name="candidate",
    description=f"An agent that simulates a candidate for {job_position} position.",
    input_func=input
)

career_coach=AssistantAgent(
    name="Career_Coach",
    model_client=az_model_client,
    description=f"An AI agnet that provides feedback and advice to candidates for a {job_position} position.",
    system_message=f'''
    You are a career coach specializing in preparing candidates for {job_position} interviews.
    Provide constructive feedback on the candidate's responses and suggest improvements.
    After the interview, summarize the candidate's performance and provide actionable advice.
    Make it under 100 words.
'''
)

terminate_condition= TextMentionTermination(text="TERMINATE")

team=RoundRobinGroupChat(
    participants=[interviewer, candidate, career_coach],
    termination_condition=terminate_condition,
    max_turns=20
)  

stream=team.run_stream(task="Conduction an interview for a Software Engineer position for SDE-2 role at Google")

async def main():
 await Console(stream)


if __name__=="__main__":
   import asyncio
   asyncio.run(main())
