
# LLM Evaluation Pipeline

This tool evaluates Large Language Model (LLM) responses for **Relevance** and **Factual Accuracy** (Hallucinations). It is designed to be scalable, efficient, and cost-effective.

## Features

- **Parallel Evaluation**: Uses `asyncio` to run evaluation tasks concurrently.
- **Provider Support**: Supports **OpenAI** (e.g., GPT-4o-mini) and **Google Gemini** (e.g., Gemini 1.5 Flash).
- **Mock Mode**: Includes a comprehensive mock mode for testing without API changes.
- **Metrics**: detailed reporting on:
  - **Latency**: User-to-AI response time.
  - **Cost**: Estimated token cost (mock pricing model).
  - **Quality**: Scores (1-10) for Relevance and Accuracy with reasoning.

## Local Setup Instructions

1.  **Clone the Repository**: Ensure you have the source code on your local machine.
2.  **Prerequisites**:
    - Python 3.8 or higher installed.
    - An OpenAI API Key or Google Gemini API Key.
3.  **Environment Setup**:
    ```powershell
    # Create a virtual environment
    python -m venv venv
    
    # Activate the virtual environment (Windows)
    .\venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```
4.  **Configuration**:
    - Create a file named `.env` in the root directory.
    - Add your keys:
      ```env
      OPENAI_API_KEY="sk-..."
      GEMINI_API_KEY="AIza..."
      ```

## Architecture

The evaluation pipeline follows a modular, asynchronous architecture designed for high throughput.

### 1. Input Layer (Validation)
- **Component**: `Pydantic` Models (`ChatConversation`, `ContextData`).
- **Function**: Ingests raw JSON data and strictly validates the schema before processing. This ensures that malformed data is caught immediately, preventing runtime errors deep in the pipeline.

### 2. Processing Layer (Async Engine)
- **Component**: `asyncio` Event Loop + `llm_judge` function.
- **Function**:
    - **Deterministic Metrics**: Latency and Cost are calculated locally on the CPU using dataset timestamps and token counts. This incurs *zero* network overhead.
    - **LLM Evaluation**: Relevance and Factual Accuracy checks are dispatched as **independent asynchronous tasks**. This means both evaluations happen simultaneously, reducing total execution time to roughly the duration of the slowest single request.

### 3. Output Layer (Reporting)
- **Component**: `Report` Model.
- **Function**: Aggregates results from all parallel tasks into a standardized JSON report.

## Design Rationale

**Why Pydantic?**
We chose Pydantic over manual JSON parsing to ensure type safety and data integrity. In data-heavy pipelines, schema drift is a common issue; Pydantic catches these issues at the entry point.

**Why Asyncio?**
Network I/O (calling OpenAI/Gemini) is the biggest bottleneck. Synchronous execution would double the latency (Relevance Time + Hallucination Time). Asyncio allows us to run them in parallel (Max(Relevance, Hallucination)), significantly speeding up processing.

**Why Stateless Script?**
The script acts as a pure function (JSON in -> JSON out). This makes it easy to containerize (Docker), run as a Lambda function, or integrate into larger CI/CD pipelines without side effects.

## Scalability Strategy

To scale this solution to **millions of daily conversations**, we have implemented several key strategies to minimize latency and cost:

1.  **Asynchronous Execution**:
    We leverage `asyncio` to prevent blocking. The pipeline does not wait for one LLM call to finish before starting the next. This maximizes throughput per worker instance.

2.  **Lightweight "Judge" Models**:
    The system is configured to use faster, cheaper models like **GPT-4o-mini** or **Gemini 1.5 Flash** as evaluators. These models are orders of magnitude cheaper and faster than reasoning models (like GPT-4o or Claude 3.5 Sonnet) while still being sufficient for grading tasks.

3.  **Local Metric Calculation**:
    Metrics like **Response Latency** and **Estimated Cost** are computed mathematically on the local CPU (using timestamps and token counts). We do not use LLMs to "guess" these values. This saves token costs and eliminates network calls for operational metrics.

4.  **Sampling Strategy (Production Recommendation)**:
    At the scale of millions, it is often unnecessary and cost-prohibitive to perform qualitative LLM evaluation on *every* single interaction. A production deployment should:
    - Compute **Latency** and **Token Cost** for **100%** of traffic (cheap, local).
    - Perform **LLM-based Quality Evaluation** (Relevance/Accuracy) on a **statistically significant sample** (e.g., 5% random sample or flagged conversations).

## Usage

### Basic Command
```powershell
python evaluate_llm.py --chat "mock_data\sample-chat-conversation-01.json" --context "mock_data\sample_context_vectors-01.json" --provider openai
```

### Mock Mode (No API Cost)
```powershell
python evaluate_llm.py --chat "mock_data\sample-chat-conversation-01.json" --context "mock_data\sample_context_vectors-01.json" --provider mock
```
