import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

import os
from dotenv import load_dotenv

# Must load env vars BEFORE any module that reads them at import time
load_dotenv(os.path.join("..", ".env"), override=True)

from datetime import datetime


from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent

from tools.file_tools import ls, read_file, write_file
from utils.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
    CODE_SANDBOX_USAGE_INSTRUCTIONS,
)
from tools.research_tools import tavily_search, think_tool, get_today_str
from tools.code_sandbox import run_code_in_sandbox
from utils.state import DeepAgentState
from tools.task_tool import _create_task_tool
from tools.todo_tools import write_todos, read_todos

# ── Azure OpenAI – main agent model ───────────────────────────────────────────
model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
)

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Tools
sub_agent_tools = [tavily_search, think_tool, run_code_in_sandbox]
built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

# Create research sub-agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["tavily_search", "think_tool", "run_code_in_sandbox"],
}

# Create task tool to delegate tasks to sub-agents
task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)

delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools

# Build prompt
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=datetime.now().strftime("%a %b %-d, %Y"),
)

INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# CODE SANDBOX USAGE\n"
    + CODE_SANDBOX_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

agent = create_agent(
    model, all_tools, system_prompt=INSTRUCTIONS, state_schema=DeepAgentState
)

