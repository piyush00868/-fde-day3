import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

WORKSPACE = Path("workspace").resolve()

# Create workspace if it doesn't exist
WORKSPACE.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECURITY
# ============================================================

def safe_path(path: str) -> Path:
    """
    Make sure the LLM can only access files
    inside our workspace directory.
    """

    requested_path = (WORKSPACE / path).resolve()

    try:
        requested_path.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError("Access outside workspace is not allowed.")

    return requested_path


# ============================================================
# TOOL 1: LIST FILES
# ============================================================

@tool
def list_files() -> str:
    """
    List all files inside the project workspace.
    Use this to understand the existing project structure.
    """

    files = []

    for path in WORKSPACE.rglob("*"):

        if path.is_file():

            relative_path = path.relative_to(WORKSPACE)

            files.append(str(relative_path))

    if not files:
        return "Workspace is empty."

    return "\n".join(files)


# ============================================================
# TOOL 2: READ FILE
# ============================================================

@tool
def read_file(path: str) -> str:
    """
    Read the contents of a file inside the workspace.
    """

    try:

        file_path = safe_path(path)

        if not file_path.exists():
            return f"File does not exist: {path}"

        if not file_path.is_file():
            return f"{path} is not a file."

        return file_path.read_text(encoding="utf-8")

    except Exception as e:

        return f"Error reading file: {e}"


# ============================================================
# TOOL 3: WRITE FILE
# ============================================================

@tool
def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a file inside the workspace.
    Automatically creates missing directories.
    """

    try:

        file_path = safe_path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path.write_text(
            content,
            encoding="utf-8"
        )

        return f"Successfully wrote {path}"

    except Exception as e:

        return f"Error writing file: {e}"


# ============================================================
# TOOL 4: RUN COMMAND
# ============================================================

@tool
def run_command(command: str) -> str:
    """
    Run a shell command inside the workspace.

    Only use this for development commands such as:
    npm install
    npm run build
    npm run dev
    """

    try:

        # Basic command protection
        forbidden = [
            "rm -rf",
            "del /f",
            "rmdir /s",
            "format",
            "shutdown",
            "reboot",
            "diskpart"
        ]

        command_lower = command.lower()

        for blocked in forbidden:

            if blocked in command_lower:

                return (
                    f"Command blocked for safety: {command}"
                )

        result = subprocess.run(
            command,
            cwd=WORKSPACE,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout

        if result.stderr:
            output += "\n" + result.stderr

        if result.returncode != 0:

            return (
                f"Command failed with exit code "
                f"{result.returncode}\n\n"
                f"{output}"
            )

        return output or "Command executed successfully."

    except subprocess.TimeoutExpired:

        return "Command timed out after 120 seconds."

    except Exception as e:

        return f"Error running command: {e}"


# ============================================================
# ALL TOOLS
# ============================================================

tools = [
    list_files,
    read_file,
    write_file,
    run_command
]


tool_map = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an autonomous coding agent.

You help the user create and modify websites.

You have access to these tools:

- list_files
- read_file
- write_file
- run_command

IMPORTANT RULES:

1. Inspect the workspace before modifying an existing project.

2. Use read_file when you need to understand existing code.

3. Use write_file to actually create or modify files.

4. Do NOT simply give the user code that they have to copy.
   Use write_file to save the code yourself.

5. After creating or modifying a project, use run_command
   when appropriate to test or build the project.

6. If a command fails, inspect the error and fix the relevant
   files yourself.

7. Work inside the provided workspace only.

8. Do not access files outside the workspace.

9. When creating a website, create all necessary files yourself.

10. Keep working until the requested task is completed.

11. At the end, briefly tell the user what you created or changed.
"""


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(user_message: str):

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", user_message)
    ]

    while True:

        response = llm_with_tools.invoke(messages)

        # Add LLM response to conversation
        messages.append(response)

        # --------------------------------------------
        # No more tools needed
        # --------------------------------------------

        if not response.tool_calls:

            print("\nAI:")
            print(response.content)

            break

        # --------------------------------------------
        # Execute every requested tool
        # --------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(
                f"\n🔧 Tool: {tool_name}"
            )

            print(
                f"📦 Arguments: {tool_args}"
            )

            selected_tool = tool_map.get(tool_name)

            if not selected_tool:

                result = f"Unknown tool: {tool_name}"

            else:

                try:

                    result = selected_tool.invoke(
                        tool_args
                    )

                except Exception as e:

                    result = f"Tool error: {e}"

            print(
                f"📤 Result:\n{result[:3000]}"
            )

            # Give result back to LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                }
            )


# ============================================================
# CHAT LOOP
# ============================================================

print("=" * 60)
print("🤖 AI WEBSITE BUILDER")
print("=" * 60)

print(
    f"\nWorkspace: {WORKSPACE}"
)

print(
    "\nType 'exit' to quit.\n"
)


while True:

    user_input = input("You: ")

    if user_input.lower() in [
        "exit",
        "quit"
    ]:

        print("\nGoodbye!")
        break

    if not user_input.strip():
        continue

    run_agent(user_input)