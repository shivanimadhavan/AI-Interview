import os
import json
from dotenv import load_dotenv
load_dotenv()
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager
)
import sys
print(sys.executable)

import autogen

#--------------------------------------------LLM declaration-------------------------------------------------


deployment_json = os.environ.get("AZURE_DEPLOYMENT_DEFAULTS")
if deployment_json:
    dd = json.loads(deployment_json)
    # IMPORTANT: use DEPLOYMENT NAME here (not the model name)
    deployment = dd.get("deployment_names", {}).get("gpt-4.1", "gpt-4.1")
else:
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]


llm_config = {
    "model": deployment,    
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],   
    "api_type": "azure",
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
}

#--------------------------------------------Agents declaration-------------------------------------------------

job_position = "Software Engineer"

interviewer = AssistantAgent(
    name="Interviewer",
    llm_config=llm_config,
    system_message=f"""
You are a professional interviewer for a {job_position} position.
Ask one clear question at a time and wait for the candidate response.
Ignore the career coach's responses while interviewing.

Ask exactly 3 questions covering:
1) technical skills and experience
2) problem-solving abilities
3) cultural fit

Keep each question under 50 words.
After the 3rd question, end with: TERMINATE
""".strip(),
)

candidate = UserProxyAgent(
    name="Candidate",
    human_input_mode="ALWAYS",
    code_execution_config={
         "use_docker": False,
    },
    description=f"A candidate interviewing for a {job_position} role.",
    llm_config=llm_config
)

career_coach = AssistantAgent(
    name="Career_Coach",
    llm_config=llm_config,
    system_message=f"""
You are a career coach specializing in {job_position} interviews.
After each candidate answer, provide brief constructive feedback.
After the interview ends, summarize performance + actionable advice.
Keep feedback under 100 words.
""".strip(),
)

groupchat = GroupChat(
    agents=[interviewer, candidate, career_coach],
    messages=[],                         
    max_round=20,                        
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,             
)

#--------------------------------------------Glue code -------------------------------------------------

manager.initiate_chat(
        interviewer,
        message="Start the interview now. Remember to end with 'TERMINATE' after your third question.",
)