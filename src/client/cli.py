""" reading from stdin with input(),
printing to stdout"""

import asyncio
import selectors
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from client.mcp_v1_chatBot import MCP_ChatBot
from log_setup import setup_logging

logger = setup_logging("RAG-Chatbot", "debug.log")


async def get_resource(chatbot: MCP_ChatBot, resource_uri):
    session = chatbot.sessions.get(resource_uri)

    # try any paper resource session
    if not session and resource_uri.startswith("papers://"):
        for uri, sess in chatbot.sessions.items():
            if uri.startswith("papers://"):
                session = sess
                break

    if not session:
        print(f"Resource '{resource_uri}' not found.")
        return

    try:
        result = await session.read_resource(uri=resource_uri)
        if result and result.contents:
            print(f"\nResource: {resource_uri}")
            print("Content:")
            print(result.contents[0].text)
        else:
            print("No content available.")

    except Exception as e:
        print(f"Error while getting resources: {e}")


async def list_prompts(chatbot: MCP_ChatBot):
    """Lists all available prompts."""
    if not chatbot.available_prompts:
        print("No prompts avaiable. ")
        return

    print("\nAvailable Prompts:")
    for prompt in chatbot.available_prompts:
        print(f"- {prompt['name']}: {prompt['description']}")
        if prompt['arguments']:
            for arg in prompt['arguments']:
                arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                print(f" - {arg_name}")


async def execute_prompt(chatbot: MCP_ChatBot, prompt_name, args):
    """Execute the prompt with the given arguments."""
    session = chatbot.sessions.get(prompt_name)
    if not session:
        print(f"Prompt '{prompt_name}' not found.")
        return
    try:
        result = await session.get_prompt(prompt_name, arguments=args)
        if result and result.messages:
            prompt_content = result.messages[0].content

        # Extract content from content
        if isinstance(prompt_content, str):
            text = prompt_content
        elif hasattr(prompt_content, 'text'):
            text = prompt_content.text
        else:
            text = " ".join(item.text if hasattr(item, 'text') else str(item) for item in prompt_content)

        print(f"\nExecuting prompt '{prompt_name}'...")
        await process_query(chatbot, text)

    except Exception as e:
        print(f"Error while executing prompt : {e}")


async def process_query(chatbot: MCP_ChatBot, query: str):
    logger.info(f"--- START PROCESS_QUERY: {query} ---")

    config = {"configurable": {"thread_id": chatbot.thread_id}}

    logger.info("Invoking Graph...")
    result = await chatbot.app.ainvoke(
        {
            "original_query": query,
            "current_query": query,
            "messages": chatbot.messages,
            "retry_count": 0,
            "clarification_question": None,
            "clarification_options": [],
        },
        config
    )

    while "__interrupt__" in result:
        logger.info("Graph Interrupted: Waiting for user clarification.")
        question_data = result["__interrupt__"][0].value

        if isinstance(question_data, dict):
            print(f"AI: {question_data.get('question', 'Could you clarify?')}")
            if question_data.get("options"):
                print("Possible meanings:", ", ".join(question_data["options"]))
        else:
            print(f"AI: {question_data}")

        # Get human clarification
        answer = (await asyncio.to_thread(input, "\nQuery:")).strip()

        # Resume graph execution with the user's answer
        logger.info(f"Resuming graph with: {answer}")
        result = await chatbot.app.ainvoke(Command(resume=answer), config)

    all_messages = result.get("messages", chatbot.messages)
    # Keep only human/AI conversation turns — no tool call noise
    chatbot.messages = [
        m for m in all_messages
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ]
    chatbot.messages = chatbot.messages[-20:]  # keep last 20 messages only

    final_response = result.get('draft_answer')
    logger.info("Graph finished execution.")

    if final_response:
        print(f"\nAI: {final_response}")

    # NOTE: pre-existing bug, carried over unchanged — t0 is started here,
    # AFTER the response above was already printed, so this always reports
    # ~0.00s regardless of how long the graph actually took. Flagging it
    # rather than fixing it here: fixing it means moving this line to the
    # top of the function, which changes behavior (the printed number),
    # not just structure — exactly the kind of change this refactor pass
    # is deliberately not making. Worth a follow-up if you want real timing.
    t0 = time.time()

    print(f"[timing] agent response took {time.time() - t0:.2f}s")


async def chat_loop(chatbot: MCP_ChatBot):
    """ Run an interactive chat loop"""
    print("\n MCP Chatbot Started !!!!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = (await asyncio.to_thread(input, "\nQuery: ")).strip()

            if query.lower() == "quit":
                break

            if not query:
                continue

            # Check for resource syntax first
            if query.startswith('@'):
                # Remove @ sign
                topic = query[1:]
                if topic == "folders":
                    resource_uri = "papers://folders"
                else:
                    resource_uri = f"papers://{topic}"
                await get_resource(chatbot, resource_uri)
                continue

            # Check for /command syntax
            if query.startswith('/'):
                parts = query.split()
                command = parts[0].lower()

                if command == '/clear':
                    chatbot.messages = []
                    print("Conversation history cleared.")
                    continue

                if command == '/prompts':
                    await list_prompts(chatbot)
                elif command == '/prompt':
                    if len(parts) < 2:
                        print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                        continue

                    prompt_name = parts[1]
                    args = {}

                    for arg in parts[2:]:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            args[key] = value

                    await execute_prompt(chatbot, prompt_name, args)
                else:
                    print(f"Unknown command: {command}")
                continue

            await process_query(chatbot, query)
            print("\n")

        except Exception as e:
            import traceback
            traceback.print_exc()


async def main():
    print("CHATBOT")
    chatbot = MCP_ChatBot()
    try:
        t0 = time.time()
        await chatbot._connect_with_retry()
        print(f"[timing] server connection took {time.time() - t0:.2f}s")
        await chatbot._build_agent_and_graph()
        await chat_loop(chatbot)
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(
        asyncio.DefaultEventLoopPolicy()
    )
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())