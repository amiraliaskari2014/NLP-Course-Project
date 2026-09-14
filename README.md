# PoliMillionaire Poliglot

Multi-domain AI agent for the PoliMillionaire quiz game. The project combines local language models, retrieval-augmented generation, live web/news search, speech transcription, symbolic math tools, and timed game automation to answer multiple-choice questions across all PoliMillionaire competitions.

The main notebook is:

```text
full/PoliMillionaire_Poliglot.ipynb
```

## Project Overview

PoliMillionaire is a timed multiple-choice quiz environment. Each question has four possible answers and must be answered within a short time window. This project builds a specialized answering pipeline for each question category instead of relying on one generic prompt for everything.

High-level flow:

```text
Start game session
        |
Receive question
        |
Choose category-specific pipeline
        |
Use model, retrieval, search, speech, or tools
        |
Select A/B/C/D
        |
Submit option ID to the game API
        |
Log result and score
```

## Covered Competitions

| ID | Category | Main Strategy |
|---:|---|---|
| 0 | Entertainment | Prompting, Wikipedia RAG, DuckDuckGo/Wikipedia hybrid RAG, ensemble voting |
| 1 | Ancient History & Politics | Advanced Wikipedia RAG, BM25, sentence embeddings, cross-encoder reranking, direct logit scoring |
| 2 | Science & Nature | Prompting, Wikipedia/DuckDuckGo retrieval, ensemble voting |
| 3 | Maths | Qwen Math model with optional SymPy/statistics tools |
| 4 | Philosophy & Psychology | Prompting and ensemble-style answering |
| 5 | News & Current Events | Serper Google News search, article scraping, Bing RSS backup, FAISS semantic retrieval |

## Repository Structure

```text
.
+-- README.md
+-- full/
|   +-- PoliMillionaire_Poliglot.ipynb     # Main combined notebook
+-- NLP_assignment_api_client/
|   +-- millionaire_client/                # Client for competitions, auth, game, leaderboard
+-- old_codes/                             # Earlier experiments and backup notebooks
```

## Main Models

### Shared General Model

```text
Qwen/Qwen2.5-7B-Instruct
```

Used for:

- zero-shot, few-shot, and chain-of-thought answering
- query construction
- direct option scoring
- tool routing/planning
- general RAG answer generation

The shared Qwen 7B model is loaded once using 4-bit BitsAndBytes quantization:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

### Math Model

```text
Qwen/Qwen2.5-Math-1.5B-Instruct
```

Used only for the Maths competition. On GPU it is loaded in `float16`; on CPU it falls back to `float32`. It is not 4-bit quantized in the current setup.

### Speech Model

```text
OpenAI Whisper
```

Default model size:

```text
turbo
```

Fallback sizes:

```text
medium -> small -> base
```

Whisper is used for speech-mode competitions. It transcribes the audio question and answer options, then passes the cleaned text into the normal text pipeline.

## Installation

The notebook was designed for a Colab-style GPU runtime, but the project can also run locally with the correct dependencies.

Core installation commands used in the notebook:

```bash
pip install -q transformers accelerate bitsandbytes sentencepiece sympy wikipedia-api
pip install -q torch --index-url https://download.pytorch.org/whl/cu118
pip install -q python-terrier sentence-transformers scikit-learn
pip install -q protobuf latex2sympy2 faiss-cpu trafilatura requests beautifulsoup4 openai-whisper
```

If running locally on a different CUDA version, install the PyTorch build that matches your system from the official PyTorch instructions.

## Required Configuration

The notebook needs access to:

- the PoliMillionaire API server
- PoliMillionaire username/password
- optional Serper API key for News search
- optional Google Drive path for saving run logs and speech audio

Recommended environment variables:

```bash
export POLI_MILLIONAIRE_API_URL="your_api_url"
export POLI_MILLIONAIRE_USERNAME="your_username"
export POLI_MILLIONAIRE_PASSWORD="your_password"
export SERPER_API_KEY="your_serper_api_key"
```

Do not commit real credentials or API keys to the repository.

## Running The Project

Open:

```text
full/PoliMillionaire_Poliglot.ipynb
```

Run the notebook in this order:

1. Setup and dependency installation
2. Client import and login
3. Shared model loading
4. Optional Whisper loading for speech mode
5. The desired competition pipeline
6. The corresponding game run cell

Example text-mode game call:

```python
play_full_game(
    competition_id=0,
    answer_fn=answer_ensemble,
    label="Entertainment ensemble",
    mode="text",
)
```

Example speech-mode game call:

```python
play_full_game(
    competition_id=0,
    answer_fn=answer_ensemble,
    label="Entertainment ensemble speech",
    mode="speech",
)
```

## Pipeline Details

### 1. Entertainment, Science, Philosophy

These sections use several prompt-based and lightweight retrieval methods:

- zero-shot prompting
- few-shot prompting
- chain-of-thought prompting
- simple Wikipedia RAG
- DuckDuckGo + Wikipedia hybrid RAG
- ensemble majority vote

Simple Wikipedia RAG flow:

```text
Question + options
        |
Build search query
        |
Wikipedia Search API
        |
Take top result
        |
Fetch page extract
        |
Pass extract directly to Qwen
        |
Return final letter
```

This first RAG section does not use chunking, BM25, embeddings, FAISS, or cross-encoder reranking. It delegates ranking to Wikipedia's own search API and uses the top returned page.

DuckDuckGo hybrid RAG flow:

```text
Question + options
        |
Build query
        |
DuckDuckGo Instant Answer API
        |
If empty, Wikipedia fallback
        |
Pass short context directly to Qwen
        |
Return final letter
```

DuckDuckGo is used as a live summary source. It returns `AbstractText` or a related-topic snippet. The code does not chunk or rerank DuckDuckGo results.

### 2. Ancient History & Politics

The History pipeline is the most advanced RAG system in the project.

Full flow:

```text
Question
   |
Direct Qwen logit scoring
   |
High confidence?
   | yes
   v
Submit direct answer

If low confidence:
   |
Build Wikipedia queries
   |
Search Wikipedia
   |
Custom candidate ranking
   |
Fetch top documents
   |
Split into chunks
   |
BM25 retrieval with PyTerrier
   |
Sentence embedding / cross-encoder reranking
   |
Score answer options against evidence
   |
Reconcile direct answer vs evidence answer
   |
Submit final option
```

Document ranking uses a custom heuristic score:

```text
candidate score =
  question keyword overlap
+ option keyword overlap
+ core named-entity overlap
+ Wikipedia search-rank bonus
- generic-title penalties
- missing-core-term penalties
```

Chunk ranking uses:

1. PyTerrier BM25
2. sentence embedding reranking
3. optional cross-encoder reranking
4. TF-IDF or lexical overlap fallback if BM25 fails

Option scoring uses:

- lexical keyword overlap
- sentence embedding similarity
- exact option text match
- option term coverage
- supporting sentence score
- topic relevance multiplier

The final answer is selected by reconciling the direct model's answer with evidence-based option scores.

### 3. News & Current Events

The News pipeline uses live news retrieval because model memory may be outdated.

Primary flow:

```text
Question
   |
Qwen generates short search keywords
   |
Optional date filter
   |
Serper Google News search
   |
Scrape top article bodies
   |
Build evidence context
   |
Ask Qwen to answer from evidence
```

Serper is not a search engine itself; it is an API service that returns Google Search or Google News style results.

The primary Serper path does not chunk documents. It passes a single evidence context to the model containing:

- title
- date
- snippet/summary
- up to a capped amount of scraped full article text

Backup flow:

```text
No clear Serper answer
   |
Bing News RSS search
   |
Scrape article text
   |
Split into chunks
   |
Embed chunks
   |
FAISS nearest-neighbor search
   |
Pass best chunks to Qwen
```

FAISS is used only in the backup path. It performs fast vector similarity search over article chunk embeddings using `IndexFlatL2`.

### 4. Maths

The Maths pipeline combines Qwen Math with optional symbolic tools.

Main flow:

```text
Question
   |
If USE_TOOL=True:
   |
Planner Qwen chooses tool or none
   |
Run SymPy/statistics tool
   |
Pass calculator result to Qwen Math
   |
Qwen Math selects A/B/C/D

If tools disabled or no usable tool:
   |
Qwen Math solves directly
   |
Extract boxed final answer
```

Available tool-router choices:

| Tool | Purpose |
|---|---|
| `sympy_simplify` | Simplify/evaluate expressions, radicals, fractions, powers |
| `sympy_solve` | Solve equations/systems, roots, sums/products/differences |
| `sympy_calculus` | Derivatives, integrals, limits, critical points |
| `sympy_matrix` | Characteristic polynomials, eigenvalues, trace, determinant |
| `probability_stats` | Expected value, binomial, hypergeometric, Bayes, combinations, statistics |
| `none` | Conceptual questions where a tool is not useful |

The math solver is constrained by token and time limits:

```python
THINK_TOKENS = 480
WRAPUP_TOKENS = 20
TOKEN_LIMIT = THINK_TOKENS + WRAPUP_TOKENS
TIME_LIMIT = 29.5
WRAPUP_TIME = 27.0
```

This gives the model time to reason while forcing a final boxed answer before the 30-second game deadline.

### 5. Speech Mode

Speech mode works by converting audio to text before using the same answer pipelines.

Flow:

```text
Fetch question audio
        |
Fetch option audio
        |
Save WAV files
        |
Whisper transcription
        |
Clean transcript noise
        |
Update question/options text
        |
Run normal category pipeline
```

The transcript cleaner removes common Whisper artifacts such as filler sounds, repeated laughter, option prefixes, and hallucinated phrases like "thanks for watching".

## Evaluation

The notebook includes mean-score calculations for each competition type. Example stored History scores:

```python
hist_amounts = [1024000, 0, 512000, 2000, 300, 512000, 0, 1000, 200, 100]
```

The evaluation section computes filtered means to reduce the impact of extreme outliers.

## Important Design Choices

### Shared Model Loading

The general Qwen model is loaded once and reused:

```text
answer_model = planner_model = shared Qwen 7B
```

This avoids accidentally loading multiple 7B models into GPU memory.

### Domain-Specific Pipelines

Different question categories need different methods:

- Entertainment often benefits from short web summaries and album/cast/track context.
- History benefits from structured Wikipedia retrieval and evidence scoring.
- News requires live search and article scraping.
- Maths benefits from exact symbolic tools.
- Speech mode requires transcription before answering.

### Timed-Game Awareness

Several parts of the project are designed around the game timer:

- model warmup before starting
- short generation limits
- Wikipedia request deadlines
- fallback answers when time is low
- direct-answer shortcuts when confidence is high

## Known Limitations

- Simple Wikipedia and DuckDuckGo RAG sections do not chunk or rerank context.
- DuckDuckGo's free Instant Answer API can return empty results for specific trivia.
- Serper ranking is trusted directly in the primary News path.
- Article scraping may fail on sites with paywalls, heavy JavaScript, or anti-bot protections.
- SymPy tools depend on the planner producing valid structured tool input.
- Speech mode accuracy depends on audio quality and available GPU/CPU memory.

## Safety And Credentials

This project should be run with credentials stored in environment variables or notebook secrets. Avoid hard-coding:

- PoliMillionaire username/password
- Serper API key
- Hugging Face tokens
- private paths or account details

Recommended pattern:

```python
import os

username = os.getenv("POLI_MILLIONAIRE_USERNAME")
password = os.getenv("POLI_MILLIONAIRE_PASSWORD")
serper_key = os.getenv("SERPER_API_KEY")
```

## Short Project Summary

PoliMillionaire Poliglot is a multi-strategy AI quiz agent. It uses one shared Qwen 7B model for general reasoning and planning, a Qwen Math model for mathematical questions, Whisper for speech input, Wikipedia/DuckDuckGo/Serper/Bing for retrieval, BM25 and semantic reranking for History, FAISS for News backup retrieval, and SymPy for exact mathematical computation.

The result is a complete competition system that can answer text and speech questions across Entertainment, History, Science, Maths, Philosophy/Psychology, and News under timed game conditions.

## Contributors
Amirali Askari
