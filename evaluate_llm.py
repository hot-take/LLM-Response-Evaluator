"""
LLM Evaluation Pipeline

SCALABILITY STRATEGY:
To ensure latency and costs remain at a minimum for real-time evaluations at scale (millions of conversations):

1.  **Asynchronous Execution**: We use `asyncio` to execute independent evaluation tasks (Relevance vs. Hallucination) in parallel. This prevents the total latency from being the sum of all serial LLM calls.
2.  **Lightweight Models**: The pipeline is designed to use faster, cheaper "Flash" or "Mini" models (e.g., Gemini-1.5-Flash, GPT-4o-mini) for the evaluator role, rather than expensive reasoning models.
3.  **Local Metrics**: Latency and Cost are calculated deterministically on the CPU without external API calls, incurring zero network latency or extra cost.
4.  **Message Sampling**: In a production environment, we would interpret "at scale" by only fully evaluating a statistically significant sample (e.g., 5%) of conversations, while running cheap metrics (Latency/Cost) on 100%.

Usage:
    python evaluate_llm.py --chat path/to/chat.json --context path/to/context.json --provider gemini
"""

import asyncio
import json
import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# --- Configuration & Imports ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Hardware/Provider imports (fail gracefully)
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Pydantic Models ---

class Message(BaseModel):
    role: str
    message: str
    created_at: str

class ChatConversation(BaseModel):
    conversation_turns: List[Message] = Field(default_factory=list)

class VectorItem(BaseModel):
    id: Union[str, int]
    text: str = ""
    tokens: int = 0

class ContextSources(BaseModel):
    vectors_used: List[Union[str, int]] = Field(default_factory=list)

class ContextInner(BaseModel):
    vector_data: List[VectorItem] = Field(default_factory=list)
    sources: ContextSources

class ContextData(BaseModel):
    data: ContextInner

class EvaluationScore(BaseModel):
    score: int
    reasoning: str
    hallucinations: Optional[List[str]] = None

class Metrics(BaseModel):
    latency_seconds: float
    estimated_cost_usd: float
    relevance: EvaluationScore
    factual_accuracy: EvaluationScore

class Report(BaseModel):
    timestamp: str
    metrics: Metrics

# --- Logic ---

def parse_iso_time(ts: str) -> datetime:
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts)

def compute_latency(turns: List[Message]) -> float:
    """Calculate time difference between last User query and AI response."""
    # Find last AI turn
    ai_turn = None
    ai_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].role in ['AI/Chatbot', 'assistant', 'model']:
            ai_turn = turns[i]
            ai_idx = i
            break
    
    if not ai_turn or ai_idx == 0:
        return 0.0

    # Find preceding User turn
    user_turn = None
    for i in range(ai_idx - 1, -1, -1):
        if turns[i].role in ['User', 'user']:
            user_turn = turns[i]
            break
    
    if not user_turn:
        return 0.0

    try:
        t_ai = parse_iso_time(ai_turn.created_at)
        t_user = parse_iso_time(user_turn.created_at)
        return (t_ai - t_user).total_seconds()
    except Exception:
        return 0.0

def compute_cost(chat: ChatConversation, context: ContextData) -> float:
    """Estimate input/output token cost (mock pricing)."""
    # Constants (Mock Pricing for 'At Scale' Estimation - e.g. $0.15/1M input)
    PRICE_IN = 0.00000015  # Per token
    PRICE_OUT = 0.00000060 # Per token

    used_ids = context.data.sources.vectors_used
    context_tokens = sum(v.tokens for v in context.data.vector_data if v.id in used_ids)
    
    chat_text = " ".join([m.message for m in chat.conversation_turns[:-1]]) # History
    chat_tokens = len(chat_text.split()) * 1.3 # Rough est
    
    last_msg = chat.conversation_turns[-1].message if chat.conversation_turns else ""
    output_tokens = len(last_msg.split()) * 1.3

    return round((context_tokens + chat_tokens) * PRICE_IN + output_tokens * PRICE_OUT, 6)

async def llm_judge(provider: str, prompt: str, model_name: str) -> EvaluationScore:
    """Generic Async LLM Judge."""
    
    # MOCK MODE (If no provider/key)
    if provider == 'mock':
        await asyncio.sleep(0.5) # Simulate latency
        return EvaluationScore(score=8, reasoning="Mock evaluation: Valid response.")

    if provider == 'openai' and AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return EvaluationScore.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            return EvaluationScore(score=0, reasoning=f"OpenAI Error: {e}")

    if provider == 'gemini' and genai:
        try:
            # Note: GenAI Python SDK async support is limited in some versions, 
            # wrapping sync call in to_thread for true parallelism if needed, 
            # but standard generate_content_async is available in newer versions.
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Using 'gemini-1.5-flash' (or user provided) for speed/cost
            model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
            
            # Gemini JSON mode enforcement via prompt + config
            res = await model.generate_content_async(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return EvaluationScore.model_validate_json(res.text)
        except Exception as e:
            return EvaluationScore(score=0, reasoning=f"Gemini Error: {e}")

    return EvaluationScore(score=0, reasoning="Provider not configured")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "gemini"])
    parser.add_argument("--model", help="Override evaluator model name")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()

    # 1. Load & Validate Inputs (Pydantic)
    try:
        with open(args.chat, 'r', encoding='utf-8') as f:
            chat_data = ChatConversation.model_validate_json(f.read())
        with open(args.context, 'r', encoding='utf-8') as f:
            context_data = ContextData.model_validate_json(f.read())
    except ValidationError as e:
        print(f"Input Validation Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File Not Found: {e}")
        sys.exit(1)

    # Prepare Data
    last_ai = next((t for t in reversed(chat_data.conversation_turns) if t.role in ['AI/Chatbot', 'assistant']), None)
    if not last_ai:
        print("No AI turn found to evaluate.")
        sys.exit(1)
        
    user_query_turn = next((t for t in chat_data.conversation_turns if t.role in ['User', 'user']), Message(role="u", message="", created_at=""))
    # (Simplified: typically you'd find the user turn immediately preceding the AI turn)

    used_vectors = [v.text for v in context_data.data.vector_data if v.id in context_data.data.sources.vectors_used]
    context_text = "\n\n".join(used_vectors)

    # 2. Parallel Evaluation (Async)
    print("Starting evaluations...")
    
    # Prompts
    rel_prompt = f"""
    Evaluate Relevance. Output JSON with fields: score (1-10), reasoning.
    Query: {user_query_turn.message}
    Response: {last_ai.message}
    """
    
    hall_prompt = f"""
    Evaluate Factual Accuracy. Output JSON with fields: score (1-10), reasoning, hallucinations (list of strings).
    Context: {context_text[:10000]}
    Response: {last_ai.message}
    """

    # Launch tasks
    task_relevance = llm_judge(args.provider, rel_prompt, args.model)
    task_hallucination = llm_judge(args.provider, hall_prompt, args.model)
    
    # Compute deterministic metrics locally
    latency = compute_latency(chat_data.conversation_turns)
    cost = compute_cost(chat_data, context_data)

    # Await LLM results
    relevance, hallucination = await asyncio.gather(task_relevance, task_hallucination)

    # 3. Report
    metrics = Metrics(
        latency_seconds=latency,
        estimated_cost_usd=cost,
        relevance=relevance,
        factual_accuracy=hallucination
    )
    
    report = Report(
        timestamp=datetime.now().isoformat(),
        metrics=metrics
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report.model_dump_json(indent=2))
    
    print(f"Success! Report saved to {args.output}")
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
