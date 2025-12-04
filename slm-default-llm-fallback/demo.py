import os
import re
import asyncio
import logging
from typing import AsyncIterable, Any, MutableSequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from agent_framework import (
    ChatAgent, 
    ChatMessage, 
    WorkflowBuilder, 
    Role,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    ChatOptions,
    AgentRunUpdateEvent, 
    AgentExecutorResponse
)
from agent_framework.azure import AzureOpenAIChatClient
from mlx_lm.utils import load
from mlx_lm.generate import generate, stream_generate

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("agent_framework").setLevel(logging.ERROR) 

load_dotenv()

class ConfidenceResult(BaseModel):
    score: int = Field(alias="confidence")
    
    @classmethod
    def parse_from_text(cls, text: str) -> "ConfidenceResult":
        match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return cls(confidence=int(match.group(1)))
        return cls(confidence=0)

class MLXChatClient(BaseChatClient):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading Local Model: {model_path}...")
        self.model, self.tokenizer = load(model_path) #type: ignore
    
    def _prepare_prompt(self, messages: list[ChatMessage]) -> str:
        msg_dicts = []
        for m in messages:
            msg_dicts.append({"role": str(m.role.value), "content": m.text or ""})
        
        # Inject instruction into the last message
        if msg_dicts:
            msg_dicts[-1]["content"] += "\nIMPORTANT: End response with 'CONFIDENCE: X' (1-10). If you are sure of your answer, you MUST output a score of 8 or higher."

        return self.tokenizer.apply_chat_template(msg_dicts, tokenize=False, add_generation_prompt=True) #type: ignore

    async def _inner_get_response(self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any) -> ChatResponse:
        prompt = self._prepare_prompt(list(messages))
        response_text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=response_text)], model_id="phi-4-mini")

    async def _inner_get_streaming_response(self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any) -> AsyncIterable[ChatResponseUpdate]:
        prompt = self._prepare_prompt(list(messages))
        generation_stream = stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=300)
        
        for response_chunk in generation_stream:
            yield ChatResponseUpdate(role=Role.ASSISTANT, text=response_chunk.text, model_id="phi-4-mini")
            await asyncio.sleep(0)

def should_fallback_to_cloud(message: AgentExecutorResponse) -> bool:
    text = message.agent_run_response.text or ""
    result = ConfidenceResult.parse_from_text(text)
    
    print(f"\n\n   📊 Verifier Score: {result.score}/10")
    
    if result.score < 8:
        print("   ⚠️ Low Confidence. Routing to Cloud...")
        return True
    
    print("   ✅ High Confidence. Workflow Complete.")
    return False

async def main():
    print("====================================================")
    print("   Cascade Pattern with Microsoft Agent Framework")
    print("====================================================\n")

    mlx_client = MLXChatClient("mlx-community/Phi-4-mini-instruct-4bit")
    
    aoi_resource = os.getenv("AZURE_OPENAI_RESOURCE")
    aoi_endpoint = f"https://{aoi_resource}.openai.azure.com" if aoi_resource and not aoi_resource.startswith("http") else aoi_resource

    azure_client = AzureOpenAIChatClient(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=aoi_endpoint, 
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-06-01"
    )

    queries = [
        # 1. Easy Fact (High Confidence)
        "What is the capital of France?",
        
        # 2. Logic/Code (High Confidence)
        "Convert this list to a JSON array: Apple, Banana, Cherry",
        
        # 3. Amiguous
        "Where is the city of Springfield located?",

        # 4. Hallucination Trap
        "Explain in 2 sentences the role of quantum healing in modeling proteins.",
        
        # 5. Reasoning
        "If I have a cabbage, a goat, and a wolf, and I need to cross a river but can only take one item at a time, and I can't leave the goat with the cabbage or the wolf with the goat, how do I do it?",
    ]

    for q in queries:
        print(f"\n❔ Query: {q}")
        print("-" * 40)
        
        # Agents hold conversation history. Creating them new here ensures they are fresh
        slm_agent = ChatAgent(
            name="Local_SLM",
            instructions="You are a helpful assistant.",
            chat_client=mlx_client
        )

        llm_agent = ChatAgent(
            name="Cloud_LLM",
            instructions="You are a fallback expert. The previous assistant was unsure. Provide a complete answer.",
            chat_client=azure_client
        )

        builder = WorkflowBuilder()
        builder.set_start_executor(slm_agent)
        
        builder.add_edge(
            source=slm_agent,
            target=llm_agent,
            condition=should_fallback_to_cloud
        )
        
        workflow = builder.build()

        current_agent = None
        
        async for event in workflow.run_stream(q):
            if isinstance(event, AgentRunUpdateEvent):
                if event.executor_id != current_agent:
                    if current_agent: print() 
                    current_agent = event.executor_id
                    print(f"   🤖 {current_agent}: ", end="", flush=True)
                
                if event.data and event.data.text:
                    print(event.data.text, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())