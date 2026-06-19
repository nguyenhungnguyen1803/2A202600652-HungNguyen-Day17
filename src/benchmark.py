from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = 0
    for exp in expected:
        # Match clean lowercase substrings
        if exp.lower() in ans_lower:
            matches += 1
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.
    Considers recall score and output length (shorter, concise answers are preferred).
    """
    recall = recall_points(answer, expected)
    if recall == 0.0:
        return 0.1
    
    # Check word count of the answer (shorter is better for quality, but must contain the info)
    words = len(answer.split())
    if words < 5:
        length_penalty = 0.5
    elif words > 40:
        length_penalty = 0.7  # Lan man
    else:
        length_penalty = 1.0
        
    return recall * length_penalty


def run_agent_benchmark(agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall_score = 0.0
    total_quality_score = 0.0
    total_compactions = 0
    
    # Track initial memory sizes
    user_ids = {c["user_id"] for c in conversations}
    initial_sizes = {}
    for uid in user_ids:
        # Clear files before starting benchmark to make it clean
        if agent_name == "Advanced":
            profile_path = agent.profile_store.path_for(uid)
            if profile_path.exists():
                profile_path.unlink()
        initial_sizes[uid] = 0

    # Feed all conversations
    question_count = 0
    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        turns = conv["turns"]
        
        # 1. Feed turns in sequence (same thread)
        for turn_idx, turn in enumerate(turns):
            if agent_name == "Baseline":
                res = agent.reply(user_id, thread_id, turn)
            else:
                res = agent.reply(user_id, thread_id, turn)
                
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]
            
        # Accumulate compactions
        total_compactions += agent.compaction_count(thread_id)

        # 2. Ask recall questions in a FRESH thread
        recall_thread_id = f"{thread_id}-recall"
        for q in conv["recall_questions"]:
            question = q["question"]
            expected = q["expected_contains"]
            
            if agent_name == "Baseline":
                res = agent.reply(user_id, recall_thread_id, question)
            else:
                res = agent.reply(user_id, recall_thread_id, question)
                
            # Accumulate recall query tokens
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]
            
            # Score
            score = recall_points(res["response"], expected)
            quality = heuristic_quality(res["response"], expected)
            
            total_recall_score += score
            total_quality_score += quality
            question_count += 1

    # Measure memory growth
    final_memory_growth = 0
    if agent_name == "Advanced":
        for uid in user_ids:
            final_size = agent.memory_file_size(uid)
            final_memory_growth += max(0, final_size - initial_sizes[uid])
            
    avg_recall = total_recall_score / question_count if question_count > 0 else 1.0
    avg_quality = total_quality_score / question_count if question_count > 0 else 1.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=final_memory_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table of benchmark results."""
    headers = [
        "Agent", 
        "Agent tokens only", 
        "Prompt tokens processed", 
        "Cross-session recall", 
        "Response quality", 
        "Memory growth (bytes)", 
        "Compactions"
    ]
    
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))
    
    lines = [header_line, sep_line]
    for r in rows:
        row_str = (
            f"{r.agent_name} | "
            f"{r.agent_tokens_only} | "
            f"{r.prompt_tokens_processed} | "
            f"{r.recall_score * 100:.1f}% | "
            f"{r.response_quality * 100:.1f}% | "
            f"{r.memory_growth_bytes} | "
            f"{r.compactions}"
        )
        lines.append(row_str)
        
    return "\n".join(lines)


def run_benchmark_suite(config) -> dict[str, str]:
    """Run benchmark suites and return formatted markdown tables."""
    # Load datasets
    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")
    
    # Initialize agents
    baseline_agent = BaselineAgent(config, force_offline=True)
    advanced_agent = AdvancedAgent(config, force_offline=True)
    
    # Standard Benchmark
    std_baseline = run_agent_benchmark("Baseline", baseline_agent, std_convs, config)
    std_advanced = run_agent_benchmark("Advanced", advanced_agent, std_convs, config)
    std_table = format_rows([std_baseline, std_advanced])
    
    # Reset/Fresh agents for Stress Benchmark
    baseline_agent_stress = BaselineAgent(config, force_offline=True)
    advanced_agent_stress = AdvancedAgent(config, force_offline=True)
    
    stress_baseline = run_agent_benchmark("Baseline", baseline_agent_stress, stress_convs, config)
    stress_advanced = run_agent_benchmark("Advanced", advanced_agent_stress, stress_convs, config)
    stress_table = format_rows([stress_baseline, stress_advanced])
    
    return {
        "standard": std_table,
        "stress": stress_table,
        "raw_results": {
            "std_baseline": std_baseline,
            "std_advanced": std_advanced,
            "stress_baseline": stress_baseline,
            "stress_advanced": stress_advanced
        }
    }


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    print("=== RUNNING MEMORY BENCHMARKS ===")
    results = run_benchmark_suite(config)
    
    print("\n--- STANDARD BENCHMARK ---")
    print(results["standard"])
    
    print("\n--- LONG-CONTEXT STRESS BENCHMARK ---")
    print(results["stress"])


if __name__ == "__main__":
    main()
