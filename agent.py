from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()


class Agent:
    """A simple AI agent that can use tools to answer questions in a multi-turn conversation"""

    def __init__(self, tools):
        self.client = OpenAI(
            base_url="https://s61m7rcraigxdazd.us-east-1.aws.endpoints.huggingface.cloud/v1/",
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = "tgi"
        self.system_message = "You are a helpful assistant that breaks down problems into steps and solves them systematically."
        self.messages = []
        self.tools = tools
        self.tool_map = {tool.get_schema()["function"]["name"]: tool for tool in tools}
        self.tool_schemas = [tool.get_schema() for tool in tools]

    def chat(self, message):
        """Process a user message and return a response"""

        # Store user input in short-term memory
        if isinstance(message, list):
            # Handle tool results
            self.messages.extend(message)
        else:
            # Handle regular user message
            self.messages.append({"role": "user", "content": message})

        # Prepare messages with system message
        messages_with_system = [{"role": "system", "content": self.system_message}] + self.messages

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            tools=self.tool_schemas if self.tools else None,
            messages=messages_with_system,
            temperature=0.1,
        )

        # Store assistant's response in short-term memory
        assistant_message = response.choices[0].message
        self.messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })

        return response


class CalculatorTool():
    """A tool for performing mathematical calculations"""

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs basic mathematical calculations, use also for simple additions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2+2', '10*5')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

    def execute(self, expression):
        """
        Evaluate mathematical expressions.
        WARNING: This tutorial uses eval() for simplicity but it is not recommended for production use.

        Args:
            expression (str): The mathematical expression to evaluate
        Returns:
            float: The result of the evaluation
        """
        try:
            result = eval(expression)
            return {"result": result}
        except:
            return {"error": "Invalid mathematical expression"}
        
        
        

def run_agent(user_input, max_turns=10):
  calculator_tool = CalculatorTool()
  agent = Agent(tools=[calculator_tool])

  i = 0

  while i < max_turns: # It's safer to use max_turns rather than while True
    i += 1
    print(f"\nIteration {i}:")

    print(f"User input: {user_input}")
    response = agent.chat(user_input)

    message = response.choices[0].message
    if message.content:
        print(f"Agent output: {message.content}")

    # Handle tool use if present
    if message.tool_calls:

        # Process all tool uses in the response
        tool_results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            print(f"Using tool {tool_name} with input {tool_input}")

            # Execute the tool
            tool = agent.tool_map[tool_name]
            tool_result = tool.execute(**tool_input)

            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
            print(f"Tool result: {tool_result}")

        # Add tool results to conversation
        user_input = tool_results
    else:
      return message.content

  return
