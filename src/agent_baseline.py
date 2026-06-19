from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).
    - Within-session memory only
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        # If API key is not present or offline is forced, use offline path
        if self.force_offline or not self.config.model.api_key:
            return self._reply_offline(thread_id, message)
        
        # Live path
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Simple offline behavior:
        - Store message in session
        - Generate a reply based ONLY on current thread
        - Update token counts
        """
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        
        session = self.sessions[thread_id]

        # Calculate prompt context: all previous messages in this session
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        # Append user message
        session.messages.append({"role": "user", "content": message})

        # Answer generation using ONLY current session memory
        ans = "Tôi không nhớ thông tin này vì tôi là Baseline Agent và không có bộ nhớ dài hạn."
        
        # Simple lookup in current session messages
        msg_lower = message.lower()
        
        # Check if we can find name in current session
        if "tên" in msg_lower:
            found = False
            for m in session.messages[:-1]:
                if m["role"] == "user":
                    m_lower = m["content"].lower()
                    name_match = re.search(r"tên\s+là\s+([A-Za-z0-9_À-ỹ]+(?:\s+[A-Za-z0-9_À-ỹ]+)*)", m["content"], re.IGNORECASE)
                    if name_match:
                        ans = f"Tên của bạn là {name_match.group(1).strip()}."
                        found = True
                        break
            if not found:
                ans = "Tôi không nhớ bạn tên là gì."

        # Check for other facts in current session
        elif "đồ uống" in msg_lower or "uống" in msg_lower:
            for m in session.messages[:-1]:
                if m["role"] == "user" and "cà phê sữa đá" in m["content"].lower():
                    ans = "Đồ uống yêu thích của bạn là cà phê sữa đá."
                    break
        elif "nơi ở" in msg_lower or "ở đâu" in msg_lower:
            for m in reversed(session.messages[:-1]):
                if m["role"] == "user":
                    if "huế" in m["content"].lower():
                        ans = "Hiện tại bạn đang ở Huế."
                        break
                    elif "đà nẵng" in m["content"].lower():
                        ans = "Hiện tại bạn đang ở Đà Nẵng."
                        break

        reply_tokens = estimate_tokens(ans)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": ans})

        return {
            "response": ans,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Invoke the live LangChain agent model."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        # Assemble prompt history
        messages_to_send = []
        for m in session.messages:
            if m["role"] == "user":
                messages_to_send.append(("user", m["content"]))
            else:
                messages_to_send.append(("assistant", m["content"]))
        messages_to_send.append(("user", message))

        # Invoke model
        chat_model = build_chat_model(self.config.model)
        response = chat_model.invoke(messages_to_send)
        ans = response.content

        # Update session
        prompt_tokens = sum(estimate_tokens(content) for role, content in messages_to_send[:-1])
        reply_tokens = estimate_tokens(ans)

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": ans})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += reply_tokens

        return {
            "response": ans,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _maybe_build_langchain_agent(self) -> None:
        """Wire `create_agent` or setup langchain baseline if needed."""
        # For baseline, standard chat model is sufficient.
        pass
