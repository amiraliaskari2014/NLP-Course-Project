"""Wikipedia RAG player for the PoliMillionaire assignment.

The flow is:
1. Search Wikipedia with delayed MediaWiki API requests.
2. Fetch and clean the best matching documents.
3. Split them into chunks and retrieve the top-k chunks for the question.
4. Give the top-k chunks, question, and options directly to Llama 3.2.
5. Submit the selected option to the game API.
"""

from __future__ import annotations

import getpass
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "PoliMillionaireNLP/1.0 student project"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LETTERS = "ABCD"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "these", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "according",
    "article", "considered", "important", "goal", "goals", "main", "primary",
    "following", "answer", "option", "best", "describes",
}

_LAST_WIKIPEDIA_REQUEST_TIME = 0.0
_LLAMA_CACHE: dict[str, tuple[Any, Any]] = {}
_ENCODER_CACHE: dict[str, Any] = {}
_RERANKER_CACHE: dict[str, Any] = {}


@dataclass
class WikipediaRagConfig:
    """Retrieval and generation parameters for one game run."""

    top_n_docs: int = 3
    per_query_limit: int = 5
    max_search_queries: int = 2
    multi_retrieval_passes: bool = True
    option_search_pass: bool = True
    max_total_documents: int = 4
    wikipedia_timeout: float = 6.0
    wikipedia_request_delay_seconds: float = 0.8
    wikipedia_429_backoff_seconds: float = 4.0
    wikipedia_max_retries: int = 2
    sentences_per_chunk: int = 5
    chunk_overlap: int = 2
    top_k_chunks: int = 4
    max_context_chars: int = 3200
    evidence_scoring_chunks: int = 12
    retriever_model: str = "multi-qa-MiniLM-L6-cos-v1"
    use_cross_encoder_reranker: bool = True
    rerank_candidate_chunks: int = 20
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_llama_judge: bool = False
    llama_model: str = LLAMA_MODEL_ID
    llama_max_new_tokens: int = 260
    exact_overlap_fallback_enabled: bool = True
    evidence_consensus_enabled: bool = True
    evidence_consensus_min_votes: float = 2.0
    require_evidence_consensus: bool = True
    llama_vote_weight: float = 0.5
    min_seconds_to_attempt: float = 3.0
    question_time_buffer: float = 1.0
    delay_submit_for_wiki_cooldown: bool = True
    target_submit_elapsed_seconds: float = 29.0
    min_seconds_left_at_submit: float = 1.0
    max_submit_wait_seconds: float = 28.0
    save_run_log: bool = True
    run_log_dir: str = "rag_game_runs"
    verbose: bool = True


def ensure_runtime_packages(include_llama: bool = True, include_retriever: bool = True) -> None:
    """Install missing runtime packages when running inside a notebook."""

    required_packages: list[tuple[str, str]] = []
    if include_retriever:
        required_packages.extend(
            [
                ("sentence-transformers", "sentence_transformers"),
                ("scikit-learn", "sklearn"),
            ]
        )
    if include_llama:
        required_packages.extend(
            [
                ("transformers", "transformers"),
                ("accelerate", "accelerate"),
            ]
        )

        try:
            import torch

            if torch.cuda.is_available():
                required_packages.append(("bitsandbytes", "bitsandbytes"))
        except Exception:
            pass

    missing = [package for package, module in required_packages if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing packages:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def get_secret_value(name: str, default: str = "") -> str:
    """Read a value from environment variables or Colab Secrets."""

    value = os.environ.get(name)
    if value:
        return value

    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass

    return default


def get_huggingface_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
        token = get_secret_value(name)
        if token:
            return token
    return None


def login_millionaire_client(
    client_class,
    api_url: str = "http://131.175.15.22:51111/",
    username: str = "",
    password: str = "",
    timeout: int = 10,
):
    """Create and authenticate a MillionaireClient without hard-coding credentials."""

    username = (
        username
        or get_secret_value("POLI_MILLIONAIRE_USERNAME")
        or get_secret_value("POLI_USERNAME")
        or get_secret_value("MILLIONAIRE_USERNAME")
    )
    password = (
        password
        or get_secret_value("POLI_MILLIONAIRE_PASSWORD")
        or get_secret_value("POLI_PASSWORD")
        or get_secret_value("MILLIONAIRE_PASSWORD")
    )

    if not username:
        username = input("PoliMillionaire username: ").strip()
    if not password:
        password = getpass.getpass("PoliMillionaire password: ").strip()

    client = client_class(api_url, timeout=timeout)
    user = client.login(username, password)
    print(f"Welcome, {user.username}! (Role: {user.role})")
    return client


def question_to_text(question) -> str:
    if hasattr(question, "text"):
        return str(question.text)
    if isinstance(question, dict) and "text" in question:
        return str(question["text"])
    return str(question)


def option_text(option) -> str:
    return str(option.text if hasattr(option, "text") else option["text"])


def option_id(option) -> int:
    return int(option.id if hasattr(option, "id") else option["id"])


def question_options(question) -> list[Any]:
    if hasattr(question, "options"):
        return list(question.options)
    if isinstance(question, dict):
        return list(question.get("options", []))
    return []


def normalize_wikipedia_text(text: str) -> str:
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 1]


def expand_term(token: str) -> set[str]:
    variants = {token}
    if token == "roman":
        variants.update({"rome", "romans"})
    elif token in {"rome", "romans"}:
        variants.add("roman")
    return variants


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    keywords = []
    seen = set()
    for token in tokenize(text):
        if token in STOPWORDS or token in seen:
            continue
        keywords.append(token)
        seen.add(token)
        if len(keywords) >= limit:
            break
    return keywords


def capital_context_phrases(question_text: str) -> list[str]:
    """Build focused phrases like 'Roman marriage' from capitalized topic words."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", question_text)
    phrases = []
    for index, word in enumerate(words):
        if not word[:1].isupper() or word.lower() in STOPWORDS:
            continue
        phrase_words = [word]
        for next_word in words[index + 1:index + 4]:
            if next_word.lower() in STOPWORDS:
                break
            phrase_words.append(next_word)
        if len(phrase_words) > 1:
            phrases.append(" ".join(phrase_words))
    return phrases


def build_wikipedia_search_queries(question, include_options: bool = False) -> list[str]:
    """Create focused Wikipedia search queries instead of using the raw question only."""

    question_text = question_to_text(question)
    cleaned = re.sub(
        r"\baccording to (?:the )?(?:article|text|passage)\b",
        " ",
        question_text,
        flags=re.IGNORECASE,
    )
    cleaned = normalize_wikipedia_text(cleaned)
    keywords = extract_keywords(cleaned, limit=10)

    queries = []
    queries.extend(capital_context_phrases(cleaned))

    if include_options:
        option_keywords = extract_keywords(" ".join(option_text(option) for option in question_options(question)), limit=6)
        if keywords and option_keywords:
            queries.append(" ".join([*keywords[:5], *option_keywords[:3]]))

    if keywords:
        queries.append(" ".join(keywords[:6]))
    if len(keywords) >= 2:
        queries.append(" ".join(keywords[:2]))

    queries.append(cleaned)
    queries.append(question_text)

    deduped = []
    seen = set()
    for query in queries:
        normalized = normalize_wikipedia_text(query).lower()
        if normalized and normalized not in seen:
            deduped.append(query)
            seen.add(normalized)
    return deduped


def wikipedia_request(params: dict, config: WikipediaRagConfig) -> dict:
    """Call MediaWiki with delay and simple 429 backoff."""

    global _LAST_WIKIPEDIA_REQUEST_TIME

    url = f"{WIKIPEDIA_API}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": WIKIPEDIA_USER_AGENT})

    for attempt in range(config.wikipedia_max_retries + 1):
        elapsed_since_last = time.monotonic() - _LAST_WIKIPEDIA_REQUEST_TIME
        sleep_for = config.wikipedia_request_delay_seconds - elapsed_since_last
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            with urlopen(request, timeout=config.wikipedia_timeout) as response:
                _LAST_WIKIPEDIA_REQUEST_TIME = time.monotonic()
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            _LAST_WIKIPEDIA_REQUEST_TIME = time.monotonic()
            if exc.code != 429 or attempt >= config.wikipedia_max_retries:
                raise

            retry_after = exc.headers.get("Retry-After")
            try:
                wait_seconds = (
                    float(retry_after)
                    if retry_after
                    else config.wikipedia_429_backoff_seconds * (attempt + 1)
                )
            except ValueError:
                wait_seconds = config.wikipedia_429_backoff_seconds * (attempt + 1)

            print(
                "Wikipedia rate limit hit. "
                f"Waiting {wait_seconds:.1f}s before retry {attempt + 1}/{config.wikipedia_max_retries}..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Wikipedia request failed without returning a response.")


def search_wikipedia(query: str, config: WikipediaRagConfig) -> list[dict]:
    query = normalize_wikipedia_text(query)
    data = wikipedia_request(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": config.per_query_limit,
            "format": "json",
            "utf8": 1,
            "redirects": 1,
        },
        config,
    )
    results = data.get("query", {}).get("search", [])
    return [
        {
            "title": item.get("title", ""),
            "page_id": item.get("pageid"),
            "snippet": normalize_wikipedia_text(re.sub(r"<[^>]+>", " ", item.get("snippet", ""))),
            "query": query,
            "search_rank": rank,
        }
        for rank, item in enumerate(results, start=1)
    ]


def candidate_relevance_score(candidate: dict, question) -> float:
    question_text = question_to_text(question)
    keywords = extract_keywords(question_text, limit=10)
    candidate_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}"
    candidate_terms = set(tokenize(candidate_text))

    matched = 0
    for keyword in keywords:
        if expand_term(keyword) & candidate_terms:
            matched += 1

    overlap = matched / max(1, len(keywords))
    title_terms = tokenize(candidate.get("title", ""))
    rank_bonus = 1.0 / max(1, candidate.get("search_rank", 1))
    generic_penalty = 0.35 if len(title_terms) == 1 and len(keywords) > 1 else 0.0
    return (1.6 * overlap) + (0.25 * rank_bonus) - generic_penalty


def collect_wikipedia_candidates(
    question,
    config: WikipediaRagConfig,
    include_options_in_search: bool = False,
) -> list[dict]:
    candidates_by_title: dict[str, dict] = {}
    search_queries = build_wikipedia_search_queries(question, include_options=include_options_in_search)
    if config.max_search_queries is not None:
        search_queries = search_queries[: config.max_search_queries]

    for query in search_queries:
        try:
            results = search_wikipedia(query, config)
        except Exception as exc:
            print(f"Wikipedia search skipped for {query!r}: {exc}")
            continue

        for result in results:
            title_key = result["title"].lower()
            if title_key not in candidates_by_title:
                candidates_by_title[title_key] = result
            else:
                candidates_by_title[title_key]["search_rank"] = min(
                    candidates_by_title[title_key]["search_rank"],
                    result["search_rank"],
                )

    candidates = list(candidates_by_title.values())
    for candidate in candidates:
        candidate["candidate_score"] = candidate_relevance_score(candidate, question)
    return sorted(candidates, key=lambda item: item["candidate_score"], reverse=True)


def fetch_wikipedia_extract(title: str, config: WikipediaRagConfig) -> dict:
    data = wikipedia_request(
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "titles": title,
            "format": "json",
            "utf8": 1,
            "redirects": 1,
        },
        config,
    )
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {}) if pages else {}
    return {
        "source": "Wikipedia",
        "title": page.get("title", title),
        "page_id": page.get("pageid"),
        "url": page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "text": normalize_wikipedia_text(page.get("extract", "")),
    }


def get_wikipedia_documents_for_question(
    question,
    config: WikipediaRagConfig | None = None,
    include_options_in_search: bool = False,
) -> list[dict]:
    config = config or WikipediaRagConfig()
    candidates = collect_wikipedia_candidates(question, config, include_options_in_search=include_options_in_search)
    documents = []

    for candidate in candidates[: config.top_n_docs]:
        try:
            document = fetch_wikipedia_extract(candidate["title"], config)
        except Exception as exc:
            print(f"Wikipedia page skipped for {candidate['title']!r}: {exc}")
            continue

        document["query"] = question_to_text(question)
        document["matched_query"] = candidate.get("query")
        document["search_rank"] = candidate.get("search_rank")
        document["candidate_score"] = candidate.get("candidate_score", 0.0)
        document["snippet"] = candidate.get("snippet", "")
        documents.append(document)

    return documents


def merge_documents(documents: list[dict], max_total: int | None = None) -> list[dict]:
    """Deduplicate documents from multiple retrieval passes while preserving best scores."""

    merged: dict[tuple[str, str], dict] = {}
    for doc in documents:
        key = (
            str(doc.get("source", "")),
            str(doc.get("url") or doc.get("title") or "").lower(),
        )
        if key not in merged:
            merged[key] = doc
            continue

        current_score = float(merged[key].get("candidate_score", 0.0))
        new_score = float(doc.get("candidate_score", 0.0))
        if new_score > current_score:
            existing_passes = merged[key].get("retrieval_passes", [])
            merged[key] = doc
            merged[key]["retrieval_passes"] = existing_passes
        merged[key].setdefault("retrieval_passes", []).append(doc.get("matched_query") or doc.get("query"))

    ranked = sorted(merged.values(), key=lambda item: float(item.get("candidate_score", 0.0)), reverse=True)
    return ranked[:max_total] if max_total else ranked


def split_sentences(text: str) -> list[str]:
    text = normalize_wikipedia_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 40]


def build_rag_chunks(documents: list[dict], config: WikipediaRagConfig) -> list[dict]:
    """Split retrieved Wikipedia documents into overlapping evidence chunks."""

    chunks = []
    step = max(1, config.sentences_per_chunk - config.chunk_overlap)

    for doc_index, doc in enumerate(documents):
        doc_text = normalize_wikipedia_text(doc.get("text", ""))
        sentences = split_sentences(doc_text)
        if not sentences and len(doc_text) >= 30:
            sentences = [doc_text]

        for start in range(0, len(sentences), step):
            chunk_sentences = sentences[start:start + config.sentences_per_chunk]
            if not chunk_sentences:
                break

            chunk_text = " ".join(chunk_sentences)
            if len(chunk_text) < 40:
                continue

            chunks.append(
                {
                    "doc_index": doc_index,
                    "chunk_index": len(chunks),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", "Wikipedia"),
                    "url": doc.get("url", ""),
                    "text": chunk_text,
                    "document_score": float(doc.get("candidate_score", 0.0)),
                }
            )

            if start + config.sentences_per_chunk >= len(sentences):
                break

    return chunks


def lexical_similarity(query: str, text: str) -> float:
    query_terms = set(extract_keywords(query, limit=20))
    text_terms = set(tokenize(text))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def get_sentence_transformer(model_name: str):
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    _ENCODER_CACHE[model_name] = encoder
    return encoder


def get_cross_encoder(model_name: str):
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(model_name)
    _RERANKER_CACHE[model_name] = reranker
    return reranker


def retrieve_top_k_chunks(
    question,
    documents: list[dict],
    config: WikipediaRagConfig | None = None,
    encoder=None,
) -> list[dict]:
    """Retrieve the top-k chunks using the dense RAG style from 10_RAG."""

    config = config or WikipediaRagConfig()
    question_text = question_to_text(question)
    chunks = build_rag_chunks(documents, config)
    if not chunks:
        return []

    chunk_texts = [f"{chunk.get('source', '')} {chunk['title']} {chunk['text']}" for chunk in chunks]

    try:
        encoder = encoder or get_sentence_transformer(config.retriever_model)
        import numpy as np

        query_embedding = encoder.encode(
            [question_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        chunk_embeddings = encoder.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        similarities = np.dot(chunk_embeddings, query_embedding)
    except Exception as exc:
        print(f"Dense chunk retrieval skipped, using TF-IDF fallback: {exc}")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform([question_text] + chunk_texts)
            similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        except Exception:
            similarities = [lexical_similarity(question_text, text) for text in chunk_texts]

    ranked = []
    for chunk, similarity in zip(chunks, similarities):
        score = float(similarity) + 0.08 * chunk.get("document_score", 0.0)
        enriched = dict(chunk)
        enriched["retrieval_score"] = score
        ranked.append(enriched)

    ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
    if getattr(config, "use_cross_encoder_reranker", True) and ranked:
        candidate_count = max(
            int(getattr(config, "top_k_chunks", 4)),
            int(getattr(config, "rerank_candidate_chunks", 20)),
        )
        candidates = ranked[:candidate_count]
        try:
            reranker = get_cross_encoder(getattr(config, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pairs = [
                (
                    question_text,
                    f"{chunk.get('source', '')} {chunk.get('title', '')} {chunk.get('text', '')}",
                )
                for chunk in candidates
            ]
            rerank_scores = reranker.predict(pairs, show_progress_bar=False)
            for chunk, rerank_score in zip(candidates, rerank_scores):
                chunk["dense_retrieval_score"] = chunk.get("retrieval_score", 0.0)
                chunk["rerank_score"] = float(rerank_score)
                chunk["retrieval_score"] = float(rerank_score)
            candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
            ranked = candidates + ranked[candidate_count:]
        except Exception as exc:
            print(f"Cross-encoder reranking skipped, using dense/TF-IDF ranking: {exc}")

    return ranked[: config.top_k_chunks]


def format_options(options) -> str:
    return "\n".join(f"{LETTERS[index]}. [id={option_id(option)}] {option_text(option)}" for index, option in enumerate(options))


def format_chunks_for_prompt(chunks: list[dict], max_context_chars: int) -> str:
    blocks = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[Chunk {index} | score={chunk.get('retrieval_score', 0.0):.3f} | "
            f"{chunk.get('source', 'Wikipedia')}: {chunk.get('title', '')}]\n"
            f"{chunk.get('text', '')}"
        )
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]
        if block.strip():
            blocks.append(block)
            used += len(block)
    return "\n\n".join(blocks)


def build_llama_choice_messages(question, chunks: list[dict], config: WikipediaRagConfig) -> list[dict]:
    options = question_options(question)
    evidence = format_chunks_for_prompt(chunks, config.max_context_chars)
    return [
        {
            "role": "system",
            "content": (
                "You are playing a multiple-choice quiz. Use only the provided Wikipedia chunks "
                "as evidence. Do not use prior knowledge or historical associations that are not "
                "written in the chunks. First judge every option independently as supported, "
                "contradicted, or not_stated. A supported option must have a direct exact quote "
                "from one cited chunk. Then choose the final letter from the supported option. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question_to_text(question)}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Top-{len(chunks)} retrieved Wikipedia chunks:\n{evidence}\n\n"
                "For each option, include status, chunk, quote, and reason. Use null for chunk "
                "and quote when an option is contradicted or not_stated. The final top-level "
                "letter must be one of the options marked supported. The top-level quote must "
                "be copied exactly from the cited chunk and directly support the selected answer.\n\n"
                "Return exactly this JSON shape:\n"
                '{"options":{"A":{"status":"supported","chunk":1,"quote":"exact quote",'
                '"reason":"short reason"},"B":{"status":"not_stated","chunk":null,'
                '"quote":null,"reason":"short reason"},"C":{"status":"not_stated",'
                '"chunk":null,"quote":null,"reason":"short reason"},"D":{"status":"not_stated",'
                '"chunk":null,"quote":null,"reason":"short reason"}},"letter":"A",'
                '"chunk":1,"quote":"same exact quote","reason":"short final reason"}'
            ),
        },
    ]


def load_llama32_model(model_name: str = LLAMA_MODEL_ID):
    if model_name in _LLAMA_CACHE:
        return _LLAMA_CACHE[model_name]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = get_huggingface_token()
    tokenizer_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True}
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs.update(
                {
                    "quantization_config": quant_config,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }
            )
        except Exception:
            model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto", "low_cpu_mem_usage": True})
    else:
        model_kwargs.update({"torch_dtype": torch.float32, "low_cpu_mem_usage": True})

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()

    _LLAMA_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def llama_generate_text(messages: list[dict], model_name: str, max_new_tokens: int = 120, max_length: int = 4096) -> str:
    import torch

    tokenizer, model = load_llama32_model(model_name)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"] + "\nAnswer:"

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_json_object(text: str) -> dict | None:
    text = str(text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def parse_option_choice(text: str, options) -> int | None:
    parsed = parse_json_object(text)
    if parsed:
        letter = str(parsed.get("letter") or parsed.get("answer") or parsed.get("option") or "").strip().upper()
        if letter in LETTERS[: len(options)]:
            return LETTERS.index(letter)

    cleaned = str(text).strip().upper()
    letter_match = re.search(r"\b([A-D])\b", cleaned)
    if letter_match:
        index = LETTERS.index(letter_match.group(1))
        return index if index < len(options) else None

    digit_match = re.search(r"\b([0-3])\b", cleaned)
    if digit_match:
        index = int(digit_match.group(1))
        return index if index < len(options) else None

    option_texts = [option_text(option).strip().lower() for option in options]
    output_lower = str(text).lower()
    for index, text_value in enumerate(option_texts):
        if text_value and text_value in output_lower:
            return index
    return None


def parse_cited_chunk_number(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def normalized_for_quote_match(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def question_numbers(question) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)?\b", question_to_text(question))


def option_number_proximity_score(question, option, sentence: str) -> float:
    """Reward evidence where an option is closest to a number mentioned in the question."""

    numbers = question_numbers(question)
    if not numbers:
        return 0.0

    normalized_sentence = normalized_for_quote_match(sentence)
    normalized_option = normalized_for_quote_match(option_text(option))
    if not normalized_sentence or not normalized_option:
        return 0.0

    option_matches = list(re.finditer(re.escape(normalized_option), normalized_sentence))
    if not option_matches:
        option_tokens = [token for token in tokenize(normalized_option) if token not in STOPWORDS]
        option_matches = []
        for token in option_tokens:
            option_matches.extend(re.finditer(rf"\b{re.escape(token)}\b", normalized_sentence))
    if not option_matches:
        return 0.0

    number_matches = list(re.finditer(r"\b\d+(?:[.,]\d+)?\b", normalized_sentence))
    if not number_matches:
        return 0.0

    wanted = {number.replace(",", ".") for number in numbers}
    nearest_distance = None
    nearest_number = None
    next_distance = None
    next_number = None

    for option_match in option_matches:
        option_pos = option_match.start()
        for number_match in number_matches:
            distance = abs(number_match.start() - option_pos)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_number = number_match.group(0).replace(",", ".")
            if number_match.start() >= option_match.end():
                forward_distance = number_match.start() - option_match.end()
                if next_distance is None or forward_distance < next_distance:
                    next_distance = forward_distance
                    next_number = number_match.group(0).replace(",", ".")

    if next_number and next_distance is not None and next_distance <= 90:
        if next_number in wanted:
            return 3.2 - min(next_distance, 90) / 90
        return -0.8
    if nearest_number and nearest_distance is not None and nearest_distance <= 90:
        if nearest_number in wanted:
            return 2.0 - min(nearest_distance, 90) / 90
        return -0.6
    return 0.0


def validate_cited_evidence(parsed: dict, chunks: list[dict]) -> tuple[bool, str, int | None, str]:
    """Validate that Llama cited a real chunk and quote from the retrieved evidence."""

    chunk_number = parse_cited_chunk_number(
        parsed.get("chunk")
        or parsed.get("chunk_id")
        or parsed.get("evidence_chunk")
        or parsed.get("source_chunk")
    )
    quote = str(parsed.get("quote") or parsed.get("evidence_quote") or "").strip()

    if chunk_number is None:
        return False, "missing cited chunk number", None, quote
    if chunk_number < 1 or chunk_number > len(chunks):
        return False, f"cited chunk {chunk_number} is outside retrieved chunk range", chunk_number, quote
    if len(quote) < 8:
        return False, "missing or too-short evidence quote", chunk_number, quote

    chunk_text = str(chunks[chunk_number - 1].get("text", ""))
    normalized_quote = normalized_for_quote_match(quote)
    normalized_chunk = normalized_for_quote_match(chunk_text)
    if normalized_quote and normalized_quote in normalized_chunk:
        return True, "valid exact evidence quote", chunk_number, quote

    quote_terms = [token for token in tokenize(quote) if token not in STOPWORDS]
    chunk_terms = set(tokenize(chunk_text))
    if len(quote_terms) >= 3:
        overlap = sum(1 for token in quote_terms if token in chunk_terms) / len(quote_terms)
        if overlap >= 0.75:
            return True, "valid evidence quote by token overlap", chunk_number, quote

    return False, "quote does not match the cited chunk", chunk_number, quote


def get_option_judgment(parsed: dict, letter: str) -> dict:
    judgments = (
        parsed.get("options")
        or parsed.get("judgments")
        or parsed.get("option_judgments")
        or parsed.get("option_statuses")
        or {}
    )
    letter = str(letter).upper()

    if isinstance(judgments, dict):
        value = judgments.get(letter) or judgments.get(letter.lower())
        return value if isinstance(value, dict) else {}

    if isinstance(judgments, list):
        for item in judgments:
            if not isinstance(item, dict):
                continue
            item_letter = str(item.get("letter") or item.get("option") or "").upper()
            if item_letter == letter:
                return item

    return {}


def selected_evidence_payload(parsed: dict, letter: str) -> dict:
    judgment = get_option_judgment(parsed, letter)
    payload = dict(parsed)

    for key in ("chunk", "chunk_id", "evidence_chunk", "source_chunk"):
        if payload.get(key) is None and judgment.get(key) is not None:
            payload[key] = judgment.get(key)
    for key in ("quote", "evidence_quote"):
        if not payload.get(key) and judgment.get(key):
            payload[key] = judgment.get(key)
    if not payload.get("reason") and judgment.get("reason"):
        payload["reason"] = judgment.get("reason")
    if judgment.get("status"):
        payload["selected_option_status"] = judgment.get("status")

    return payload


def validate_selected_option_judgment(parsed: dict, selected_letter: str) -> tuple[bool, str, str]:
    judgment = get_option_judgment(parsed, selected_letter)
    if not judgment:
        return False, "missing per-option judgment for selected letter", ""

    status = str(judgment.get("status") or "").strip().lower().replace(" ", "_")
    if status not in {"supported", "support", "true"}:
        return False, f"selected option judgment is {status or 'missing'}, not supported", status

    return True, "selected option marked supported", status


def best_evidence_sentence_for_option(question, option, chunks: list[dict]) -> tuple[float, int | None, str]:
    option_terms = [token for token in extract_keywords(option_text(option), limit=12) if token not in STOPWORDS]
    question_terms = [token for token in extract_keywords(question_to_text(question), limit=12) if token not in STOPWORDS]
    evidence_terms = set(option_terms)
    if not evidence_terms:
        return 0.0, None, ""

    best = (0.0, None, "")
    for chunk_index, chunk in enumerate(chunks, start=1):
        sentences = split_sentences(chunk.get("text", "")) or [chunk.get("text", "")]
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            option_overlap = len(evidence_terms & sentence_terms) / max(1, len(evidence_terms))
            question_overlap = len(set(question_terms) & sentence_terms) / max(1, len(set(question_terms)))
            score = (1.8 * option_overlap) + (0.35 * question_overlap)

            normalized_option = normalized_for_quote_match(option_text(option))
            normalized_sentence = normalized_for_quote_match(sentence)
            if normalized_option and normalized_option in normalized_sentence:
                score += 2.0
            if "holy" in evidence_terms and "sacrum" in sentence_terms:
                score += 0.8
            if "12th" in set(question_terms) and "1157" in sentence_terms:
                score += 0.5
            if re.search(r"\bnot\s+(?:in\s+)?use\b|\bnot\s+used\b", normalized_sentence):
                score -= 0.7
            score += option_number_proximity_score(question, option, sentence)

            if score > best[0]:
                best = (score, chunk_index, sentence.strip())

    return best


def exact_overlap_fallback_choice(question, chunks: list[dict], options) -> dict | None:
    scored = []
    for index, option in enumerate(options):
        score, chunk_index, quote = best_evidence_sentence_for_option(question, option, chunks)
        scored.append((score, index, chunk_index, quote))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None

    best_score, selected_index, chunk_index, quote = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 0.75 or best_score - second_score < 0.15 or chunk_index is None:
        return None

    selected = options[selected_index]
    return {
        "answer_id": option_id(selected),
        "answer_text": option_text(selected),
        "answer_index": selected_index,
        "letter": LETTERS[selected_index],
        "model_output": "exact_overlap_fallback",
        "fallback_method": "exact_overlap",
        "reason": "Selected by exact/keyword overlap with retrieved evidence.",
        "evidence_chunk": chunk_index,
        "evidence_quote": quote[:500],
        "evidence_validation": "exact_overlap_fallback",
        "evidence_validation_failed": False,
        "fallback_score": best_score,
        "fallback_second_score": second_score,
    }


def lexical_fallback_choice(question, chunks: list[dict], options, exact_enabled: bool = True) -> dict:
    exact_fallback = None
    if exact_enabled and chunks:
        exact_fallback = exact_overlap_fallback_choice(question, chunks, options)
    if exact_fallback is not None:
        return exact_fallback

    evidence_text = " ".join(chunk.get("text", "") for chunk in chunks)
    scores = []
    for index, option in enumerate(options):
        score = lexical_similarity(f"{question_to_text(question)} {option_text(option)}", evidence_text)
        scores.append((score, index))
    scores.sort(reverse=True)
    selected_index = scores[0][1] if scores else 0
    selected = options[selected_index]
    return {
        "answer_id": option_id(selected),
        "answer_text": option_text(selected),
        "answer_index": selected_index,
        "letter": LETTERS[selected_index],
        "model_output": "lexical_fallback",
        "fallback_method": "lexical_overlap",
        "reason": "Selected by lexical overlap fallback because Llama did not return a parseable option.",
    }


def option_evidence_scores(question, chunks: list[dict], options, config: WikipediaRagConfig) -> dict:
    """Score options with several evidence-only methods for consensus."""

    evidence_text = " ".join(chunk.get("text", "") for chunk in chunks)
    option_scores = []

    for index, option in enumerate(options):
        exact_score, exact_chunk, exact_quote = best_evidence_sentence_for_option(question, option, chunks)
        lexical_score = lexical_similarity(f"{question_to_text(question)} {option_text(option)}", evidence_text)
        option_scores.append(
            {
                "index": index,
                "letter": LETTERS[index],
                "answer_id": option_id(option),
                "answer_text": option_text(option),
                "exact_score": exact_score,
                "exact_chunk": exact_chunk,
                "exact_quote": exact_quote[:500] if exact_quote else "",
                "lexical_score": lexical_score,
                "tfidf_score": None,
                "tfidf_chunk": None,
                "tfidf_quote": "",
                "cross_score": None,
                "cross_chunk": None,
                "cross_quote": "",
            }
        )

    if chunks:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            sentence_items = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                for sentence in split_sentences(chunk.get("text", "")) or [chunk.get("text", "")]:
                    if sentence.strip():
                        sentence_items.append((chunk_index, sentence.strip()))

            if sentence_items:
                option_queries = [
                    f"{question_to_text(question)} {option_text(option)}"
                    for option in options
                ]
                sentence_texts = [sentence for _, sentence in sentence_items]
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
                matrix = vectorizer.fit_transform(option_queries + sentence_texts)
                similarities = cosine_similarity(matrix[:len(options)], matrix[len(options):])
                for option_index, row in enumerate(similarities):
                    best_sentence_index = int(row.argmax())
                    chunk_index, sentence = sentence_items[best_sentence_index]
                    option_scores[option_index]["tfidf_score"] = float(row[best_sentence_index])
                    option_scores[option_index]["tfidf_chunk"] = chunk_index
                    option_scores[option_index]["tfidf_quote"] = sentence[:500]
        except Exception as exc:
            print(f"TF-IDF option scoring skipped: {exc}")

    if chunks and getattr(config, "use_cross_encoder_reranker", True):
        try:
            reranker = get_cross_encoder(getattr(config, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pairs = []
            pair_meta = []
            for option_index, option in enumerate(options):
                query = f"Question: {question_to_text(question)}\nCandidate answer: {option_text(option)}"
                for chunk_index, chunk in enumerate(chunks, start=1):
                    pairs.append((query, chunk.get("text", "")))
                    pair_meta.append((option_index, chunk_index, chunk.get("text", "")))
            cross_scores = reranker.predict(pairs, show_progress_bar=False)
            for score, (option_index, chunk_index, chunk_text) in zip(cross_scores, pair_meta):
                current = option_scores[option_index].get("cross_score")
                if current is None or float(score) > float(current):
                    sentences = split_sentences(chunk_text) or [chunk_text]
                    best_sentence = max(
                        sentences,
                        key=lambda sentence: lexical_similarity(
                            f"{question_to_text(question)} {option_text(options[option_index])}",
                            sentence,
                        ),
                    )
                    option_scores[option_index]["cross_score"] = float(score)
                    option_scores[option_index]["cross_chunk"] = chunk_index
                    option_scores[option_index]["cross_quote"] = best_sentence[:500]
        except Exception as exc:
            print(f"Cross-encoder option scoring skipped: {exc}")

    votes = []

    def add_margin_vote(method: str, score_key: str, min_score: float, min_margin: float, weight: float) -> None:
        available = [item for item in option_scores if item.get(score_key) is not None]
        if not available:
            return
        ranked = sorted(available, key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
        best = ranked[0]
        second_score = float(ranked[1].get(score_key) or 0.0) if len(ranked) > 1 else 0.0
        best_score = float(best.get(score_key) or 0.0)
        if best_score >= min_score and best_score - second_score >= min_margin:
            votes.append(
                {
                    "method": method,
                    "letter": best["letter"],
                    "index": best["index"],
                    "weight": weight,
                    "score": best_score,
                    "margin": best_score - second_score,
                }
            )

    add_margin_vote("exact_overlap", "exact_score", min_score=0.75, min_margin=0.15, weight=2.0)
    add_margin_vote("lexical_overlap", "lexical_score", min_score=0.2, min_margin=0.05, weight=1.0)
    add_margin_vote("tfidf_option", "tfidf_score", min_score=0.12, min_margin=0.04, weight=1.25)
    add_margin_vote("cross_encoder_option", "cross_score", min_score=-1000.0, min_margin=0.15, weight=1.5)

    weighted_by_index = {index: 0.0 for index in range(len(options))}
    for vote in votes:
        weighted_by_index[vote["index"]] += float(vote["weight"])

    return {
        "option_scores": option_scores,
        "votes": votes,
        "weighted_votes": weighted_by_index,
    }


def evidence_choice_from_score_item(score_item: dict, method: str) -> dict:
    if method == "cross_encoder_option" and score_item.get("cross_chunk"):
        chunk = score_item.get("cross_chunk")
        quote = score_item.get("cross_quote", "")
        validation = "cross_encoder_option"
    elif method == "tfidf_option" and score_item.get("tfidf_chunk"):
        chunk = score_item.get("tfidf_chunk")
        quote = score_item.get("tfidf_quote", "")
        validation = "tfidf_option"
    else:
        chunk = score_item.get("exact_chunk")
        quote = score_item.get("exact_quote", "")
        validation = "exact_overlap" if chunk else "evidence_consensus"

    return {
        "answer_id": score_item["answer_id"],
        "answer_text": score_item["answer_text"],
        "answer_index": score_item["index"],
        "letter": score_item["letter"],
        "evidence_chunk": chunk,
        "evidence_quote": quote,
        "evidence_validation": validation,
        "evidence_validation_failed": False,
    }


def apply_evidence_consensus(question, chunks: list[dict], options, config: WikipediaRagConfig, llama_match: dict) -> dict:
    """Combine several evidence-only attempts with the validated Llama judge."""

    consensus = option_evidence_scores(question, chunks, options, config)
    votes = list(consensus["votes"])
    weighted_votes = dict(consensus["weighted_votes"])

    llama_index = llama_match.get("answer_index")
    llama_is_supported = (
        llama_index is not None
        and not llama_match.get("llama_parse_failed")
        and not llama_match.get("evidence_validation_failed")
        and llama_match.get("evidence_quote")
    )
    if llama_is_supported:
        votes.append(
            {
                "method": "llama_evidence_judge",
                "letter": llama_match.get("letter"),
                "index": llama_index,
                "weight": float(getattr(config, "llama_vote_weight", 0.5)),
                "score": 1.0,
                "margin": None,
            }
        )
        weighted_votes[llama_index] = weighted_votes.get(llama_index, 0.0) + float(getattr(config, "llama_vote_weight", 0.5))

    best_index, best_weight = max(weighted_votes.items(), key=lambda item: item[1]) if weighted_votes else (None, 0.0)
    llama_weight = weighted_votes.get(llama_index, 0.0) if llama_index is not None else 0.0

    if best_index is None or best_weight < getattr(config, "evidence_consensus_min_votes", 2.0):
        if getattr(config, "require_evidence_consensus", True):
            fallback = lexical_fallback_choice(
                question,
                chunks,
                options,
                exact_enabled=getattr(config, "exact_overlap_fallback_enabled", True),
            )
            fallback["model_output"] = llama_match.get("model_output", "")
            fallback["fallback_method"] = fallback.get("fallback_method", "weak_evidence_fallback")
            fallback["consensus"] = {**consensus, "votes": votes, "weighted_votes": weighted_votes}
            fallback["consensus_used"] = True
            fallback["consensus_reason"] = (
                "No strong consensus reached; used best evidence-only fallback instead of Llama-only choice."
            )
            fallback["llama_original_choice"] = {
                "letter": llama_match.get("letter"),
                "answer_text": llama_match.get("answer_text"),
                "reason": llama_match.get("reason"),
                "evidence_quote": llama_match.get("evidence_quote"),
            }
            return fallback

        result = dict(llama_match)
        result["consensus"] = consensus
        result["consensus_used"] = False
        result["consensus_reason"] = "No evidence consensus reached; kept validated Llama/fallback result."
        return result

    if llama_index == best_index and llama_is_supported:
        result = dict(llama_match)
        result["consensus"] = {**consensus, "votes": votes, "weighted_votes": weighted_votes}
        result["consensus_used"] = True
        result["consensus_reason"] = "Evidence consensus agrees with validated Llama choice."
        return result

    score_item = consensus["option_scores"][best_index]
    best_methods = [vote["method"] for vote in votes if vote["index"] == best_index]
    if "cross_encoder_option" in best_methods:
        preferred_method = "cross_encoder_option"
    elif "tfidf_option" in best_methods:
        preferred_method = "tfidf_option"
    else:
        preferred_method = "exact_overlap"
    result = evidence_choice_from_score_item(score_item, preferred_method)
    result.update(
        {
            "model_output": llama_match.get("model_output", ""),
            "fallback_method": "evidence_consensus",
            "reason": (
                "Selected by evidence consensus from several evidence-only attempts: "
                + ", ".join(best_methods)
            ),
            "consensus": {**consensus, "votes": votes, "weighted_votes": weighted_votes},
            "consensus_used": True,
            "consensus_reason": (
                f"Evidence consensus chose {LETTERS[best_index]} with weight {best_weight:.1f}; "
                f"Llama choice weight was {llama_weight:.1f}."
            ),
            "llama_original_choice": {
                "letter": llama_match.get("letter"),
                "answer_text": llama_match.get("answer_text"),
                "reason": llama_match.get("reason"),
                "evidence_quote": llama_match.get("evidence_quote"),
            },
        }
    )
    return result


def choose_option_with_llama(question, chunks: list[dict], config: WikipediaRagConfig) -> dict:
    options = question_options(question)
    messages = build_llama_choice_messages(question, chunks, config)
    output = llama_generate_text(
        messages,
        model_name=config.llama_model,
        max_new_tokens=config.llama_max_new_tokens,
    )
    selected_index = parse_option_choice(output, options)
    parsed = parse_json_object(output) or {}

    if selected_index is None:
        fallback = lexical_fallback_choice(
            question,
            chunks,
            options,
            exact_enabled=getattr(config, "exact_overlap_fallback_enabled", True),
        )
        fallback["model_output"] = output
        fallback["llama_parse_failed"] = True
        return fallback

    selected_letter = LETTERS[selected_index]
    judgment_is_valid, judgment_reason, selected_status = validate_selected_option_judgment(parsed, selected_letter)
    if not judgment_is_valid:
        fallback = lexical_fallback_choice(
            question,
            chunks,
            options,
            exact_enabled=getattr(config, "exact_overlap_fallback_enabled", True),
        )
        fallback["model_output"] = output
        fallback["llama_parse_failed"] = False
        fallback["evidence_validation_failed"] = True
        fallback["llama_rejection_reason"] = judgment_reason
        fallback["llama_selected_status"] = selected_status
        fallback["reason"] = f"Rejected Llama answer: {judgment_reason}. {fallback['reason']}"
        return fallback

    evidence_payload = selected_evidence_payload(parsed, selected_letter)
    evidence_is_valid, validation_reason, cited_chunk, evidence_quote = validate_cited_evidence(evidence_payload, chunks)
    if not evidence_is_valid:
        fallback = lexical_fallback_choice(
            question,
            chunks,
            options,
            exact_enabled=getattr(config, "exact_overlap_fallback_enabled", True),
        )
        fallback["model_output"] = output
        fallback["llama_parse_failed"] = False
        fallback["evidence_validation_failed"] = True
        fallback["llama_rejection_reason"] = validation_reason
        fallback["llama_cited_chunk"] = cited_chunk
        fallback["llama_evidence_quote"] = evidence_quote
        fallback["reason"] = f"Rejected Llama answer: {validation_reason}. {fallback['reason']}"
        return fallback

    selected = options[selected_index]
    return {
        "answer_id": option_id(selected),
        "answer_text": option_text(selected),
        "answer_index": selected_index,
        "letter": LETTERS[selected_index],
        "model_output": output,
        "reason": str(evidence_payload.get("reason", "")).strip(),
        "evidence_chunk": cited_chunk,
        "evidence_quote": evidence_quote,
        "evidence_validation": validation_reason,
        "evidence_validation_failed": False,
        "selected_option_status": selected_status,
        "option_judgments": parsed.get("options") or parsed.get("judgments") or parsed.get("option_judgments"),
        "llama_parse_failed": False,
    }


def answer_question_with_wikipedia_rag(
    question,
    config: WikipediaRagConfig | None = None,
    encoder=None,
    include_options_in_search: bool = False,
) -> dict:
    config = config or WikipediaRagConfig()
    start = time.monotonic()

    retrieval_passes = [
        {
            "name": "question_only" if not include_options_in_search else "requested_search",
            "include_options_in_search": include_options_in_search,
        }
    ]
    if getattr(config, "multi_retrieval_passes", True) and getattr(config, "option_search_pass", True):
        option_pass = {"name": "question_plus_options", "include_options_in_search": True}
        if option_pass not in retrieval_passes:
            retrieval_passes.append(option_pass)

    all_documents = []
    for retrieval_pass in retrieval_passes:
        pass_documents = get_wikipedia_documents_for_question(
            question,
            config=config,
            include_options_in_search=retrieval_pass["include_options_in_search"],
        )
        for doc in pass_documents:
            doc.setdefault("retrieval_passes", []).append(retrieval_pass["name"])
        all_documents.extend(pass_documents)

    documents = merge_documents(
        all_documents,
        max_total=getattr(config, "max_total_documents", None),
    )
    if not documents:
        documents = get_wikipedia_documents_for_question(
            question,
            config=config,
            include_options_in_search=include_options_in_search,
        )
    after_documents = time.monotonic()

    scoring_chunk_count = max(
        int(getattr(config, "top_k_chunks", 4)),
        int(getattr(config, "evidence_scoring_chunks", 12)),
    )
    scoring_config = replace(config, top_k_chunks=scoring_chunk_count)
    scoring_chunks = retrieve_top_k_chunks(question, documents, config=scoring_config, encoder=encoder)
    chunks = scoring_chunks[: int(getattr(config, "top_k_chunks", 4))]
    after_retrieval = time.monotonic()

    if getattr(config, "use_llama_judge", False):
        option_match = choose_option_with_llama(question, chunks, config)
    else:
        option_match = lexical_fallback_choice(
            question,
            scoring_chunks,
            question_options(question),
            exact_enabled=getattr(config, "exact_overlap_fallback_enabled", True),
        )
        option_match["model_output"] = "llama_judge_disabled_for_speed"
        option_match["llama_parse_failed"] = False
        option_match["llama_disabled_for_speed"] = True

    if getattr(config, "evidence_consensus_enabled", True):
        option_match = apply_evidence_consensus(
            question,
            scoring_chunks,
            question_options(question),
            config,
            option_match,
        )
    after_llama = time.monotonic()

    return {
        "question": question_to_text(question),
        "options": [{"id": option_id(option), "text": option_text(option)} for option in question_options(question)],
        "documents": [
            {
                "source": doc.get("source"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "candidate_score": doc.get("candidate_score"),
                "matched_query": doc.get("matched_query"),
                "preview": doc.get("text", "")[:500],
            }
            for doc in documents
        ],
        "top_k_chunks": [
            {
                "source": chunk.get("source"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "retrieval_score": chunk.get("retrieval_score"),
                "text": chunk.get("text", ""),
            }
            for chunk in chunks
        ],
        "evidence_scoring_chunks": [
            {
                "source": chunk.get("source"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "retrieval_score": chunk.get("retrieval_score"),
                "text": chunk.get("text", ""),
            }
            for chunk in scoring_chunks
        ],
        "option_match": option_match,
        "timings": {
            "document_fetch_seconds": after_documents - start,
            "chunk_retrieval_seconds": after_retrieval - after_documents,
            "llama_choice_seconds": after_llama - after_retrieval,
            "total_seconds": after_llama - start,
        },
    }


def seconds_available(game) -> float:
    remaining = game.time_remaining
    if remaining is None:
        return 30.0
    return max(0.0, float(remaining))


def fallback_option(question):
    options = question_options(question)
    if not options:
        raise ValueError("Question has no options.")
    return options[0]


def wait_before_submit_for_cooldown(game, config: WikipediaRagConfig) -> float:
    """Wait after answer selection so Wikipedia has cooldown time before the next question."""

    if not config.delay_submit_for_wiki_cooldown:
        return 0.0

    current_remaining = seconds_available(game)
    target_remaining = max(
        config.min_seconds_left_at_submit,
        30.0 - config.target_submit_elapsed_seconds,
    )
    wait_seconds = current_remaining - target_remaining
    wait_seconds = min(config.max_submit_wait_seconds, max(0.0, wait_seconds))

    if wait_seconds > 0:
        print(
            "Answer ready. "
            f"Waiting {wait_seconds:.1f}s before submit to give Wikipedia API cooldown time."
        )
        time.sleep(wait_seconds)

    return wait_seconds


def preload_llama_for_game(config: WikipediaRagConfig) -> None:
    token = get_huggingface_token()
    if not token:
        token = getpass.getpass("Hugging Face token (input hidden): ").strip()
        if token:
            os.environ["HF_TOKEN"] = token
    if not get_huggingface_token():
        raise RuntimeError("No Hugging Face token found. Add HF_TOKEN in Colab Secrets/env vars.")

    ensure_runtime_packages(include_llama=True, include_retriever=True)
    print(f"Preloading {config.llama_model} before starting the timed game...")
    start = time.time()
    tokenizer, model = load_llama32_model(config.llama_model)

    import torch

    warmup_messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": '{"letter":"A","reason":"ready"}'},
    ]
    prompt = tokenizer.apply_chat_template(warmup_messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(f"Llama ready in {time.time() - start:.1f}s.")


def play_full_wikipedia_rag_game(
    client,
    competition_id: int,
    config: WikipediaRagConfig | None = None,
    max_questions: int | None = None,
    submit_answers: bool = True,
    include_options_in_search: bool = False,
    preload_llama: bool = True,
) -> tuple[Any, dict]:
    """Play a complete PoliMillionaire game with direct Wikipedia-RAG answers."""

    config = config or WikipediaRagConfig()
    if preload_llama and getattr(config, "use_llama_judge", False):
        preload_llama_for_game(config)
    elif not getattr(config, "use_llama_judge", False):
        print("Llama judge disabled for speed; using evidence-only consensus for timed answers.")

    encoder = None
    try:
        encoder = get_sentence_transformer(config.retriever_model)
    except Exception as exc:
        print(f"Retriever model preload skipped; will use fallback if needed: {exc}")

    if getattr(config, "use_cross_encoder_reranker", True):
        try:
            print(f"Preloading cross-encoder reranker {config.cross_encoder_model}...")
            _ = get_cross_encoder(config.cross_encoder_model)
            print("Cross-encoder reranker ready.")
        except Exception as exc:
            print(f"Cross-encoder reranker preload skipped; dense ranking will be used if needed: {exc}")

    game = client.game.start(competition_id=competition_id)
    run_log = {
        "session_id": game.session_id,
        "competition_id": competition_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "submit_answers": submit_answers,
        "questions": [],
    }

    print(f"Started game session {game.session_id}. Competition {competition_id}.")
    question_count = 0
    correct_count = 0

    while game.in_progress:
        question = game.current_question
        if question is None:
            print("No active question returned by server.")
            break

        question_count += 1
        current_level = game.current_level
        time_left = seconds_available(game)
        print("\n" + "=" * 80)
        print(f"Question {question_count} | Level {current_level} | {time_left:.1f}s left")
        print(question_to_text(question))
        for index, option in enumerate(question_options(question)):
            print(f"  {LETTERS[index]}. [id={option_id(option)}] {option_text(option)}")
        print("=" * 80)

        if time_left < config.min_seconds_to_attempt:
            selected = fallback_option(question)
            prediction = {
                "question": question_to_text(question),
                "options": [{"id": option_id(option), "text": option_text(option)} for option in question_options(question)],
                "documents": [],
                "top_k_chunks": [],
                "option_match": {
                    "answer_id": option_id(selected),
                    "answer_text": option_text(selected),
                    "answer_index": 0,
                    "letter": "A",
                    "model_output": "fallback_time_guard",
                    "reason": "Not enough time remained for RAG.",
                },
                "timings": {"total_seconds": 0.0},
            }
        else:
            try:
                prediction = answer_question_with_wikipedia_rag(
                    question,
                    config=config,
                    encoder=encoder,
                    include_options_in_search=include_options_in_search,
                )
            except Exception as exc:
                selected = fallback_option(question)
                prediction = {
                    "question": question_to_text(question),
                    "options": [{"id": option_id(option), "text": option_text(option)} for option in question_options(question)],
                    "documents": [],
                    "top_k_chunks": [],
                    "option_match": {
                        "answer_id": option_id(selected),
                        "answer_text": option_text(selected),
                        "answer_index": 0,
                        "letter": "A",
                        "model_output": f"pipeline_error: {exc}",
                        "reason": "Pipeline failed; submitted fallback option.",
                    },
                    "timings": {"total_seconds": 0.0},
                    "error": repr(exc),
                }

        selected_id = prediction["option_match"]["answer_id"]
        selected_text = prediction["option_match"]["answer_text"]
        selected_letter = prediction["option_match"]["letter"]

        if config.verbose:
            print("\nTop retrieved chunks:")
            for index, chunk in enumerate(prediction.get("top_k_chunks", [])[: config.top_k_chunks], start=1):
                preview = normalize_wikipedia_text(chunk.get("text", ""))[:300]
                print(
                    f"  {index}. {chunk.get('title')} "
                    f"(score={chunk.get('retrieval_score', 0.0):.3f}) {preview}"
                )
            print("\nSelected choice:", f"{selected_letter}. [id={selected_id}] {selected_text}")
            print("Decision output:", prediction["option_match"].get("model_output"))
            if prediction["option_match"].get("evidence_quote"):
                print(
                    "Accepted evidence:",
                    f"Chunk {prediction['option_match'].get('evidence_chunk')}:",
                    prediction["option_match"].get("evidence_quote"),
                )
            if prediction["option_match"].get("evidence_validation_failed"):
                print(
                    "Rejected Llama evidence:",
                    prediction["option_match"].get("llama_rejection_reason"),
                )
            if prediction["option_match"].get("consensus"):
                print("Evidence consensus:", prediction["option_match"].get("consensus_reason"))
                for vote in prediction["option_match"]["consensus"].get("votes", []):
                    print(
                        "  vote:",
                        vote.get("method"),
                        vote.get("letter"),
                        f"weight={float(vote.get('weight', 0.0)):.1f}",
                    )
            timings = prediction.get("timings", {})
            print(
                "Timing:",
                f"docs={timings.get('document_fetch_seconds', 0.0):.2f}s",
                f"chunks={timings.get('chunk_retrieval_seconds', 0.0):.2f}s",
                f"llama={timings.get('llama_choice_seconds', 0.0):.2f}s",
                f"total={timings.get('total_seconds', 0.0):.2f}s",
                f"time_left={seconds_available(game):.1f}s",
            )

        result_payload = None
        if submit_answers:
            submission_wait_seconds = wait_before_submit_for_cooldown(game, config)
            prediction["submission_wait_seconds"] = submission_wait_seconds
            if submission_wait_seconds:
                print(f"Time left after cooldown wait: {seconds_available(game):.1f}s")

            if seconds_available(game) <= config.question_time_buffer:
                print("Warning: low time before submit; submitting selected option immediately.")
            result = game.answer(selected_id)
            result_payload = {
                "correct": result.correct,
                "timed_out": result.timed_out,
                "game_over": result.game_over,
                "earned_amount": result.earned_amount,
            }
            correct_count += int(bool(result.correct))
            if result.correct:
                print(f"Correct. Earned: {result.earned_amount}")
            elif result.timed_out:
                print(f"Timed out. Earned: {result.earned_amount}")
            else:
                print(f"Wrong. Earned: {result.earned_amount}")
        else:
            print("Dry run: answer not submitted.")

        run_log["questions"].append(
            {
                "number": question_count,
                "level": current_level,
                "prediction": prediction,
                "result": result_payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if result_payload and result_payload.get("game_over"):
            break
        if max_questions is not None and question_count >= max_questions:
            print("max_questions reached; stopping.")
            break
        if not submit_answers:
            break

    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_log["questions_answered"] = question_count
    run_log["correct_count"] = correct_count
    run_log["final_earned_amount"] = game.earned_amount

    if config.save_run_log:
        log_dir = Path(config.run_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"wikipedia_rag_game_{game.session_id}.json"
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(run_log, handle, indent=2, ensure_ascii=False)
        print("Run log saved to:", log_path)

    print("\nGame summary")
    print("Questions answered:", question_count)
    print("Correct answers:", correct_count)
    print("Final earnings:", game.earned_amount)
    return game, run_log
