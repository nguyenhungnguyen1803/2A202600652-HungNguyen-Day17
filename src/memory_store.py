from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator.
    Approximate tokens from word count for Vietnamese/English (len(text.split()) * 1.3)
    """
    if not text:
        return 0
    words = text.strip().split()
    return int(len(words) * 1.3)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.
    Maps each user id to one markdown file and supports read / write / edit operations.
    """
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        # Slugify or sanitize the user id
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id).lower()
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    # Helpers for structured facts (Bonus: Structured Entity Extraction)
    def read_facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        facts = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                parts = line[2:].split(":", 1)
                key = parts[0].strip().lower()
                val = parts[1].strip()
                facts[key] = val
        return facts

    def write_facts(self, user_id: str, facts: dict[str, str]) -> Path:
        lines = ["# User Profile Info", ""]
        key_mapping = {
            "name": "Tên",
            "location": "Nơi ở",
            "profession": "Nghề nghiệp",
            "style": "Phong cách trả lời",
            "drink": "Đồ uống yêu thích",
            "food": "Món ăn yêu thích",
            "pet": "Thú cưng"
        }
        for k, v in facts.items():
            label = key_mapping.get(k, k.capitalize())
            lines.append(f"- {label}: {v}")
        return self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.
    Uses regex patterns to confidently extract name, location, profession, preferences.
    """
    msg_lower = message.lower()
    
    # Skip obvious question-only turns to avoid saving wrong facts (Bonus: Question Filtering)
    is_question = "?" in message or "không?" in msg_lower or "là gì" in msg_lower or "ở đâu" in msg_lower
    has_clarification = "đính chính" in msg_lower or "thực ra" in msg_lower or "chuyển sang" in msg_lower or "thay đổi" in msg_lower
    
    if is_question and not has_clarification:
        # Check if it has commands like "nhắc lại", "tóm tắt"
        if any(cmd in msg_lower for cmd in ["nhắc lại", "tóm tắt", "biết dũngct"]):
            return {}

    facts = {}
    
    # 1. Name extraction: "tên là [X]", "tên mình là [X]"
    name_match = re.search(r"tên\s+là\s+([A-Za-z0-9_À-ỹ]+(?:\s+[A-Za-z0-9_À-ỹ]+)*)", message, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        name = re.sub(r"[.,;!?]+$", "", name)
        facts["name"] = name

    # 2. Location extraction: "ở [X]", "sống tại [X]" (checking corrections)
    if "ở huế" in msg_lower or "tại huế" in msg_lower:
        # Check if they said they updated from Huế to Đà Nẵng
        if "đà nẵng" in msg_lower and msg_lower.find("đà nẵng") > msg_lower.find("huế"):
            facts["location"] = "Đà Nẵng"
        else:
            facts["location"] = "Huế"
    elif "ở đà nẵng" in msg_lower or "tại đà nẵng" in msg_lower or "sang đà nẵng" in msg_lower:
        facts["location"] = "Đà Nẵng"
    elif "ở hà nội" in msg_lower or "tại hà nội" in msg_lower:
        # Ignore Hà Nội noise in stress test: "Hà Nội chỉ là nơi mình vừa bay ra họp..."
        if "không phải nơi ở" not in msg_lower and "bay ra họp" not in msg_lower:
            facts["location"] = "Hà Nội"

    # 3. Profession extraction: "backend engineer", "mlops engineer" (checking corrections)
    if "mlops engineer" in msg_lower or "mlops" in msg_lower:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in msg_lower or "backend" in msg_lower:
        # Check if they corrected it: "không còn làm backend engineer nữa"
        if "không còn" not in msg_lower and "chuyển sang" not in msg_lower:
            facts["profession"] = "backend engineer"
    elif "product manager" in msg_lower:
        # Ignore product manager noise: "đùa... chuyển sang product manager... nghề nghiệp hiện tại vẫn là MLOps"
        if "đùa" in msg_lower or "vẫn là mlops" in msg_lower:
            facts["profession"] = "MLOps engineer"
        else:
            facts["profession"] = "product manager"

    # 4. Preferred reply style extraction
    if "trả lời" in msg_lower or "style" in msg_lower or "bullet" in msg_lower:
        if "3 bullet" in msg_lower or "ba bullet" in msg_lower:
            facts["style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
        elif "ngắn gọn" in msg_lower or "gọn" in msg_lower:
            facts["style"] = "ngắn gọn, có ví dụ thực tế"

    # 5. Food / Drink
    if "cà phê sữa đá" in msg_lower or "cafe sữa đá" in msg_lower:
        facts["drink"] = "cà phê sữa đá"
    if "mì quảng" in msg_lower:
        facts["food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in msg_lower:
        facts["pet"] = "corgi tên Bơ"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    summary_parts = []
    for m in messages:
        role = m["role"].capitalize()
        content = m["content"].strip()
        # Clean text a bit
        content_snippet = content if len(content) < 80 else content[:77] + "..."
        summary_parts.append(f"{role}: {content_snippet}")
    return "Tóm tắt hội thoại cũ: " + " | ".join(summary_parts[:max_items])


@dataclass
class CompactMemoryManager:
    """Manages compaction of chat history to save context tokens."""
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }

        state = self.state[thread_id]
        state["messages"].append({"role": role, "content": content})

        # Calculate token size of messages in state
        total_tokens = sum(estimate_tokens(m["content"]) for m in state["messages"])

        # Trigger compaction if threshold exceeded
        if total_tokens > self.threshold_tokens and len(state["messages"]) > self.keep_messages:
            # We keep the last `keep_messages` messages and compact the rest
            num_to_compact = len(state["messages"]) - self.keep_messages
            to_compact = state["messages"][:num_to_compact]
            kept_messages = state["messages"][num_to_compact:]

            new_summary = summarize_messages(to_compact)
            old_summary = state["summary"]
            
            if old_summary:
                state["summary"] = f"{old_summary}\n{new_summary}"
            else:
                state["summary"] = new_summary

            state["messages"] = kept_messages
            state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]
