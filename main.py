"""
main.py — CLI entry point for Deep Agent Coder

Usage:
    python main.py                        # interactive loop with default prompt
    python main.py "your question here"   # single prompt then exit
    streamlit run app.py                  # launch the Streamlit UI instead
"""

import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from formatter import format_messages, stream_agent
from agent.deep_agent import agent

DEFAULT_PROMPT = (
    "What happened to the president of Venezuela Nicolás Maduro in the past few days?"
)


async def ask(prompt: str):
    result = await stream_agent(
        agent,
        {"messages": [{"role": "user", "content": prompt}]},
    )
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    # Single-shot mode: prompt passed as CLI argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = asyncio.run(ask(prompt))
        if result and "messages" in result:
            format_messages(result["messages"])
        return

    # Interactive loop mode
    prompt = DEFAULT_PROMPT
    print(f"\nDeep Agent Coder  |  type 'exit' to quit\n{'─'*50}")

    while True:
        if not prompt:
            prompt = input("\nPrompt: ").strip()
            if not prompt or prompt.lower() in ("exit", "quit", "q"):
                print("Bye.")
                break

        print(f"\nUSER: {prompt}\n{'─'*50}")
        result = asyncio.run(ask(prompt))
        if result and "messages" in result:
            format_messages(result["messages"])

        prompt = ""  # next iteration will prompt for input


if __name__ == "__main__":
    main()