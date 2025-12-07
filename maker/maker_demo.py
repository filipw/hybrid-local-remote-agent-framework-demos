import os
import re
import json
import asyncio
import logging
from collections import Counter
from typing import Any, List, MutableSequence, AsyncIterable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# MLX Imports
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler

# Agent Framework Imports
from agent_framework import (
    ChatAgent, 
    ChatMessage,
    ChatResponseUpdate, 
    WorkflowBuilder, 
    WorkflowContext,
    Executor,
    handler,
    Role,
    BaseChatClient,
    ChatResponse,
    ChatOptions,
    AgentRunUpdateEvent,
    AgentExecutorResponse
)
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR) 
load_dotenv()

class MakerState(BaseModel):
    steps: List[str] = Field(default_factory=list)
    current_step_idx: int = 0
    results: List[str] = Field(default_factory=list)
    current_votes: Counter = Field(default_factory=Counter)
    attempts: int = 0
    k_threshold: int = 3
    max_attempts: int = 15
    is_complete: bool = False

class MLXStatelessClient(BaseChatClient):
    """
    STRICTLY STATELESS: Processes only the last message.
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print(f"📥 Loading Local Model: {model_path}...")
        self.model, self.tokenizer = load(model_path) #type: ignore
    
    def _prepare_prompt(self, messages: list[ChatMessage]) -> str:
        last_msg = messages[-1]
        fresh_history = [{"role": "user", "content": last_msg.text or ""}]
        return self.tokenizer.apply_chat_template(fresh_history, tokenize=False, add_generation_prompt=True) #type: ignore

    async def _inner_get_response(self, *, messages: MutableSequence[ChatMessage], **kwargs) -> ChatResponse:
        return ChatResponse(messages=[])

    async def _inner_get_streaming_response(self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any) -> AsyncIterable[ChatResponseUpdate]:
        yield ChatResponseUpdate(role=Role.ASSISTANT, text="")

    async def generate_fast(self, messages: list[ChatMessage]) -> str:
        prompt = self._prepare_prompt(messages)
        response_text = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=150, 
            verbose=False, 
            sampler=make_sampler(temp=0.5)
        )
        return response_text

class ManagerClient(BaseChatClient):
    """
    ORCHESTRATOR LOGIC: Handles the prompt construction for Step 1 vs Step N.
    """
    def __init__(self, state: MakerState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    async def _generate_text(self) -> str:
        if self.state.is_complete:
            return f"WORKFLOW_COMPLETE: {self.state.results[-1] if self.state.results else 'N/A'}"

        if self.state.current_step_idx < len(self.state.steps):
            current_step = self.state.steps[self.state.current_step_idx]
            
            if not self.state.results:
                # STEP 1: No previous result exists.
                # We command the model to solve the problem using only the data in the step.
                prompt = (
                    f"Current Task: {current_step}\n\n"
                    "INSTRUCTION: Perform the calculation described in the Task. "
                    "Do NOT look for a 'Previous Result'. Use the numbers provided in the text.\n"
                    "You MUST end your response with exactly 'Final Answer: HH:MM AM/PM'."
                )
            else:
                # STEP 2+: We have a previous result.
                # We inject it and command the model to update it.
                last_result = self.state.results[-1]
                prompt = (
                    f"Previous Result: {last_result}\n"
                    f"Current Task: {current_step}\n\n"
                    "INSTRUCTION: Update the 'Previous Result' based on the 'Current Task'.\n"
                    "You MUST end your response with exactly 'Final Answer: HH:MM AM/PM'."
                )
            return prompt
        
        return "ERROR: No steps remaining."

    async def _inner_get_response(self, *, messages: MutableSequence[ChatMessage], **kwargs) -> ChatResponse:
        text = await self._generate_text()
        return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=text)])

    async def _inner_get_streaming_response(self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any) -> AsyncIterable[ChatResponseUpdate]:
        text = await self._generate_text()
        yield ChatResponseUpdate(role=Role.ASSISTANT, text=text)

class VotingExecutor(Executor):
    def __init__(self, name: str, client: MLXStatelessClient, state: MakerState):
        super().__init__(id=name)
        self.client = client
        self.state = state

    def _extract_answer(self, text: str) -> str:
        match = re.search(r"Final Answer:\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", text, re.IGNORECASE)
        if match:
            raw_time = match.group(1).upper().replace(" ", "")
            if len(raw_time) == 6: raw_time = "0" + raw_time
            return raw_time
        return "PARSE_ERROR"

    def _get_text_content(self, message: Any) -> str:
        if isinstance(message, ChatMessage): return message.text or ""
        if hasattr(message, "agent_run_response"): return message.agent_run_response.text or ""
        return str(message)

    @handler
    async def process(self, message: object, ctx: WorkflowContext[ChatMessage]):
        input_text = self._get_text_content(message)
        
        # Visibility: Print the current task intent
        if self.state.attempts == 0 and "Current Task:" in input_text:
            task_line = input_text.split("Current Task:")[1].split("\n")[0].strip()
            print(f"\n👉 Processing Step {self.state.current_step_idx + 1}: {task_line}")

        msgs = [ChatMessage(role=Role.USER, text=input_text)]
        output_text = await self.client.generate_fast(msgs)

        ans = self._extract_answer(output_text)
        self.state.attempts += 1
        
        status_msg = ""
        if ans == "PARSE_ERROR":
            print(f"   ❌ Attempt {self.state.attempts}: Parse Error")
            status_msg = "RETRY"
        else:
            self.state.current_votes[ans] += 1
            leader, count = self.state.current_votes.most_common(1)[0]
            runner_up = 0
            if len(self.state.current_votes) > 1:
                runner_up = self.state.current_votes.most_common(2)[1][1]
            
            margin = count - runner_up
            print(f"   🎲 Attempt {self.state.attempts}: {ans} | Leader: {leader} (+{margin})")

            if margin >= self.state.k_threshold:
                print(f"   🎉 CONVERGENCE: {leader}")
                status_msg = f"RESOLVED: {leader}"
                self._commit_step(leader)
            
            elif self.state.attempts >= self.state.max_attempts:
                print(f"   ⚠️ FORCED: {leader}")
                status_msg = f"RESOLVED: {leader}"
                self._commit_step(leader)
            else:
                status_msg = "RETRY"

        await ctx.send_message(ChatMessage(role=Role.ASSISTANT, text=status_msg))

    def _commit_step(self, result: str):
        self.state.results.append(result)
        self.state.current_step_idx += 1
        self.state.current_votes.clear()
        self.state.attempts = 0
        if self.state.current_step_idx >= len(self.state.steps):
            self.state.is_complete = True

def create_transitions(state: MakerState):
    def parse_plan(response: AgentExecutorResponse) -> bool:
        text = response.agent_run_response.text or "[]"
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            state.steps = json.loads(clean_text)
            print("\n📋 DECOMPOSITION PLAN:")
            print(json.dumps(state.steps, indent=2))
            print("-" * 40)
            return True
        except: return False

    def to_solver(response: AgentExecutorResponse) -> bool: return not state.is_complete
    def to_manager(response: AgentExecutorResponse) -> bool: return True
    return parse_plan, to_solver, to_manager

async def main():
    print("====================================================")
    print("   MAKER Protocol: Final Fixed Version")
    print("====================================================\n")

    state = MakerState()
    t_parse, t_to_solver, t_to_manager = create_transitions(state)

    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="Cloud_Planner",
            instructions=(
                "You are a decomposition engine. Your goal is to break a word problem into a sequential list of atomic, self-contained calculation instructions.\n\n"
                "RULES:\n"
                "1. DATA PRESERVATION: You MUST include the specific numbers, durations, or values from the text in your instructions. Do not say 'Calculate arrival', say 'Add 45 minutes to the time'.\n"
                "2. DEPENDENCY: Assume the previous step's output is available as 'the previous result'.\n"
                "3. FORMAT: Output ONLY a raw JSON list of strings."
            ),
        ) as cloud_planner,
    ):
        manager = ChatAgent(
            name="Manager", 
            instructions="Orchestrator", 
            chat_client=ManagerClient(state)
        )
        
        mlx_client = MLXStatelessClient("mlx-community/Phi-4-mini-instruct-4bit")
        
        solver = VotingExecutor(
            name="Voting_Solver", 
            client=mlx_client, 
            state=state
        )

        builder = WorkflowBuilder()
        builder.set_start_executor(cloud_planner)
        builder.add_edge(source=cloud_planner, target=manager, condition=t_parse)
        builder.add_edge(source=manager, target=solver, condition=t_to_solver)
        builder.add_edge(source=solver, target=manager, condition=t_to_manager)

        workflow = builder.build()

        user_query = (
            "A train leaves Station A at 8:15 AM. It takes 45 minutes to reach Station B. "
            "It stops at Station B for 10 minutes. "
            "It takes 1 hour and 50 minutes to reach Station C. "
            "It stops at Station C for 15 minutes. "
            "It takes 30 minutes to reach Station D. "
            "What time does the train arrive at Station D?"
        )
        
        print(f"🚀 Query: {user_query}")

        async for event in workflow.run_stream(user_query):
            if isinstance(event, AgentRunUpdateEvent):
                if event.executor_id == "Manager" and "WORKFLOW_COMPLETE" in (event.data.text or ""):
                    print(f"\n==========================================")
                    print(f"🤖 Final Answer: {event.data.text.split(': ')[1]}")
                    print(f"📝 Correct Answer: 11:45 AM")
                    print(f"==========================================")
                    break

if __name__ == "__main__":
    asyncio.run(main())