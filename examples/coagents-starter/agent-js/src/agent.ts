/**
 * This is the main entry point for the agent.
 * It defines the workflow graph, state, tools, nodes and edges.
 */

import { z } from "zod";
import { RunnableConfig } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";

// 1. Import necessary helpers for CopilotKit actions
import {
  convertActionsToDynamicStructuredTools,
  CopilotKitStateAnnotation,
} from "@copilotkit/sdk-js/langgraph";

// 2. Define our agent state, which includes CopilotKit state to
//    provide actions to the state.
export const AgentStateAnnotation = Annotation.Root({
  proverbs: Annotation<string[]>,
  ...CopilotKitStateAnnotation.spec /* TODO figure out how to make this not any */,
});

// 3. Define the type for our agent state
export type AgentState = typeof AgentStateAnnotation.State;

// 4. Define a simple tool to get the weather statically
const getWeather = tool(
  (args) => {
    return `The weather for ${args.location} is 70 degrees.`;
  },
  {
    name: "getWeather",
    description: "Get the weather for a given location.",
    schema: z.object({
      location: z.string().describe("The location to get weather for"),
    }),
  }
);

// 5. Put our tools into an array
const tools = [getWeather];

// 6.1 Define the model, lower temperature for deterministic responses
const model =
  process.env.LLM_TYPE === "ollama"
    ? new ChatOllama({
        temperature: 0,
        model: process.env.LLM_MODEL || "qwen3:14b",
        baseUrl: process.env.LLM_BASE_URL || "http://127.0.0.1:11434",
      })
    : new ChatOpenAI({
        temperature: 0,
        model: process.env.LLM_MODEL || "gpt-4.1",
      });
// 6. Define the chat node, which will handle the chat logic
async function chat_node(state: AgentState, config: RunnableConfig) {
  console.log("Chat node state:", JSON.stringify(state, null, 2));
  // 6.2 Bind the tools to the model, include CopilotKit actions. This allows
  //     the model to call tools that are defined in CopilotKit by the frontend.
  const modelWithTools = model.bindTools!([
    ...(state.copilotkit?.actions
      ? convertActionsToDynamicStructuredTools(state.copilotkit.actions)
      : []),
    ...tools,
  ]);

  // 6.3 Define the system message, which will be used to guide the model, in this case
  //     we also add in the language to use from the state.
  const systemMessage = new SystemMessage({
    content: `/nothink You are a helpful assistant.`,
  });

  // 6.4 Invoke the model with the system message and the messages in the state
  const response = await modelWithTools.invoke(
    [systemMessage, ...state.messages],
    config
  );

  // 6.5 Return the response, which will be added to the state
  return {
    messages: response,
  };
}

// 7. Define the function that determines whether to continue or not,
//    this is used to determine the next node to run
function shouldContinue({ messages, copilotkit }: AgentState) {
  // 7.1 Get the last message from the state
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // 7.2 If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    const actions = copilotkit?.actions;
    const toolCallName = lastMessage.tool_calls![0].name;

    // 7.3 Only route to the tool node if the tool call is not a CopilotKit action
    if (!actions || actions.every((action) => action.name !== toolCallName)) {
      return "tool_node";
    }
  }

  console.log(JSON.stringify(lastMessage, null, 2));

  // 7.4 Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the workflow graph
const workflow = new StateGraph(AgentStateAnnotation)
  .addNode("chat_node", chat_node)
  .addNode("tool_node", new ToolNode(tools))
  .addEdge(START, "chat_node")
  .addEdge("tool_node", "chat_node")
  .addConditionalEdges("chat_node", shouldContinue);

const memory = new MemorySaver();

export const graph = workflow.compile({
  checkpointer: memory,
});
