# 🤖 Agents

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Implementation in BeeAI Framework](#implementation-in-beeai-framework)
  - [ReActAgent](#react-agent)
  - [Agent Execution Process](#agent-execution-process)
- [Customizing Agent Behavior](#customizing-agent-behavior)
  - [1. Setting Execution Policy](#1-setting-execution-policy)
  - [2. Overriding Prompt Templates](#2-overriding-prompt-templates)
  - [3. Adding Tools](#3-adding-tools)
  - [4. Configuring Memory](#4-configuring-memory)
  - [5. Event Observation](#5-event-observation)
- [Creating Your Own Agent](#creating-your-own-agent)
- [Agent with Memory](#agent-with-memory)
- [Agent Workflows](#agent-workflows)
- [Resources](#resources)
<!-- /TOC -->

---

## Overview

AI agents built on large language models (LLMs) provide a structured approach to solving complex problems. Unlike simple LLM interactions, agents can:

- 🔄 Execute multi-step reasoning processes
- 🛠️ Utilize tools to interact with external systems
- 📝 Remember context from previous interactions
- 🔍 Plan and revise their approach based on feedback

Agents control the path to solving a problem, acting on feedback to refine their plan, a capability that improves performance and helps them accomplish sophisticated tasks.

> 💡 **Tip:** For a deeper understanding of AI agents, read this [research article on AI agents and LLMs](https://research.ibm.com/blog/what-are-ai-agents-llm).

---

## Implementation in BeeAI Framework

### ReAct Agent

BeeAI framework provides a robust implementation of the `ReAct` pattern ([Reasoning and Acting](https://arxiv.org/abs/2210.03629)), which follows this general flow:

```
Thought → Action → Observation → Thought → ...
```

This pattern allows agents to reason about a task, take actions using tools, observe results, and continue reasoning until reaching a conclusion.

### Agent Execution Process

Let's see how a ReActAgent approaches a simple question:

**Input prompt:** "What is the current weather in Las Vegas?"

**First iteration:**
```
thought: I need to retrieve the current weather in Las Vegas. I can use the OpenMeteo function to get the current weather forecast for a location.
tool_name: OpenMeteo
tool_input: {"location": {"name": "Las Vegas"}, "start_date": "2024-10-17", "end_date": "2024-10-17", "temperature_unit": "celsius"}
```

**Second iteration:**
```
thought: I have the current weather in Las Vegas in Celsius.
final_answer: The current weather in Las Vegas is 20.5°C with an apparent temperature of 18.3°C.
```

> ⚠️ **Note:** During execution, the agent emits partial updates as it generates each line, followed by complete updates. Updates follow a strict order: first all partial updates for "thought," then a complete "thought" update, then moving to the next component.

For more complex tasks, the agent may perform many more iterations, utilizing different tools and reasoning steps.

For practical examples, see:
- [simple.py](/python/examples/agents/simple.py) - Basic example of a Bee Agent using OpenMeteo and DuckDuckGo tools
- [bee.py](/python/examples/agents/bee.py) - More complete example using Wikipedia integration
- [granite.py](/python/examples/agents/granite.py) - Example using the Granite model

---

## Customizing Agent Behavior

You can customize your agent's behavior in five ways:

### 1. Setting Execution Policy

Control how the agent runs by configuring retries, timeouts, and iteration limits.

From [bee.py](/python/examples/agents/bee.py):
```python
response = await agent.run(
    BeeRunInput(prompt=prompt),
    {
        "execution": {
            "max_retries_per_step": 3,
            "total_max_retries": 10,
            "max_iterations": 20,
        }
    },
).observe(observer)
```

> 💡 **Tip:** The default is zero retries and no timeout. For complex tasks, increasing the max_iterations is recommended.

### 2. Overriding Prompt Templates

Customize how the agent formats prompts, including the system prompt that defines its behavior.

From [agent_sys_prompt.py](/python/examples/templates/agent_sys_prompt.py):
```python
prompt = SystemPromptTemplate.render(
    SystemPromptTemplateInput(
        instructions="You are a helpful AI assistant!", 
        tools=[ToolDefinition(**tool.prompt_data())], 
        tools_length=1
    )
)
```

The agent uses several templates that you can override:
1. **System Prompt** - Defines the agent's behavior and capabilities
2. **User Prompt** - Formats the user's input
3. **Tool Error** - Handles tool execution errors
4. **Tool Input Error** - Handles validation errors
5. **Tool No Result Error** - Handles empty results
6. **Tool Not Found Error** - Handles references to missing tools
7. **Invalid Schema Error** - Handles parsing errors

### 3. Adding Tools

Enhance your agent's capabilities by providing it with tools to interact with external systems.

From [simple.py](/python/examples/agents/simple.py):
```python
agent = BeeAgent(
    bee_input=BeeInput(
        llm=llm, 
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], 
        memory=UnconstrainedMemory()
    )
)
```

**Available tools include:**
- Search tools (`DuckDuckGoSearchTool`)
- Weather tools (`OpenMeteoTool`)
- Knowledge tools (`LangChainWikipediaTool`)
- And many more in the `beeai_framework.tools` module

> 💡 **Tip:** See the [tools.md](/python/docs/tools.md) documentation for more information on available tools and creating custom tools.

### 4. Configuring Memory

Memory allows your agent to maintain context across multiple interactions.

From [simple.py](/python/examples/agents/simple.py):
```python
agent = BeeAgent(
    bee_input=BeeInput(
        llm=llm, 
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], 
        memory=UnconstrainedMemory()
    )
)
```

**Memory types for different use cases:**
- [UnconstrainedMemory](/python/examples/memory/unconstrainedMemory.py) - For unlimited storage
- [SlidingMemory](/python/examples/memory/slidingMemory.py) - For keeping only the most recent messages
- [TokenMemory](/python/examples/memory/tokenMemory.py) - For managing token limits
- [SummarizeMemory](/python/examples/memory/summarizeMemory.py) - For summarizing previous conversations

> 💡 **Tip:** See the [memory.md](/python/docs/memory.md) documentation for more information on memory types.

### 5. Event Observation

Monitor the agent's execution by observing events it emits. This allows you to track its reasoning process, handle errors, or implement custom logging.

From [simple.py](/python/examples/agents/simple.py):
```python
def update_callback(data: dict, event: EventMeta) -> None:
    print(f"Agent({data['update']['key']}) 🤖 : ", data['update']['parsedValue'])

def on_update(emitter: Emitter) -> None:
    emitter.on("update", update_callback)

output: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What's the current weather in Las Vegas?")
).observe(on_update)
```

> 💡 **Tip:** See the [emitter.md](/python/docs/emitter.md) documentation for more information on event observation.

---

## Creating Your Own Agent

To create your own agent, you must implement the agent's base class (`BaseAgent`).

From [custom_agent.py](/python/examples/memory/custom_agent.py):
```text
Coming soon
```

---

## Agent with Memory

Agents can be configured to use memory to maintain conversation context and state.

From [agentMemory.py](/python/examples/memory/agentMemory.py):
```py
def create_agent() -> BeeAgent:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    # Initialize the agent
    agent = BeeAgent(BeeInput(llm=llm, memory=memory, tools=[]))

    return agent

# Create user message
user_input = "Hello world!"
user_message = UserMessage(user_input)

# Await adding user message to memory
await memory.add(user_message)

# Run agent
response = await agent.run(
    BeeRunInput(
        prompt=user_input,
        options={
            "execution": {
                "max_retries_per_step": 3,
                "total_max_retries": 10,
                "max_iterations": 20,
            }
        },
    )
)

# Create and store assistant's response
assistant_message = AssistantMessage(response.result.text)
await memory.add(assistant_message)
```

**Memory types for different use cases:**
- [UnconstrainedMemory](/python/examples/memory/unconstrainedMemory.py) - For unlimited storage
- [SlidingMemory](/python/examples/memory/slidingMemory.py) - For keeping only the most recent messages
- [TokenMemory](/python/examples/memory/tokenMemory.py) - For managing token limits
- [SummarizeMemory](/python/examples/memory/summarizeMemory.py) - For summarizing previous conversations

---

## Agent Workflows

For complex applications, you can create multi-agent workflows where specialized agents collaborate.

From [multi_agents.py](/python/examples/workflows/multi_agents.py):
```python
workflow = AgentWorkflow(name="Smart assistant")
workflow.add_agent(
    agent=AgentFactoryInput(
        name="WeatherForecaster",
        instructions="You are a weather assistant. Respond only if you can provide a useful answer.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=BeeAgentExecutionConfig(max_iterations=3),
    )
)
workflow.add_agent(
    agent=AgentFactoryInput(
        name="Researcher",
        instructions="You are a researcher assistant. Respond only if you can provide a useful answer.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
)
workflow.add_agent(
    agent=AgentFactoryInput(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )
)

prompt = "What is the weather in New York?"
memory = UnconstrainedMemory()
await memory.add(UserMessage(content=prompt))
response = await workflow.run(messages=memory.messages)
```

**Example Workflow Patterns:**
- [multi_agents.py](/python/examples/workflows/multi_agents.py) - Multiple specialized agents working together
- [memory.py](/python/examples/workflows/memory.py) - Memory-aware workflow for conversation

> 💡 **Tip:** See the [workflows.md](/python/docs/workflows.md) documentation for more information.

---

## Resources

- **Examples:**
  - [simple.py](/python/examples/agents/simple.py) - Basic agent implementation
  - [bee.py](/python/examples/agents/bee.py) - More complete implementation
  - [granite.py](/python/examples/agents/granite.py) - Using Granite model
  - [agents.ipynb](/python/examples/notebooks/agents.ipynb) - Interactive notebook examples

- **Related Documentation:**
  - [Tools Documentation](/python/docs/tools.md)
  - [Memory Documentation](/python/docs/memory.md)
  - [Workflows Documentation](/python/docs/workflows.md)
  - [Emitter Documentation](/python/docs/emitter.md)
