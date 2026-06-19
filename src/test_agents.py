from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, ProviderConfig
from memory_store import UserProfileStore, CompactMemoryManager, estimate_tokens


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    model_config = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=None
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=60,  # Low threshold to trigger compaction quickly
        compact_keep_messages=2,      # Keep only 2 messages
        model=model_config,
        judge_model=model_config
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "test_user_01"

    # Write text
    content = "# User Info\n- Tên: Nguyễn Văn A\n- Nơi ở: Hà Nội\n"
    path = store.write_text(user_id, content)
    assert path.exists()
    assert store.file_size(user_id) > 0

    # Read text
    read_content = store.read_text(user_id)
    assert "Nguyễn Văn A" in read_content

    # Edit text
    changed = store.edit_text(user_id, "Hà Nội", "Đà Nẵng")
    assert changed
    edited_content = store.read_text(user_id)
    assert "Đà Nẵng" in edited_content
    assert "Hà Nội" not in edited_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages
    )
    thread_id = "thread_01"

    # Append short messages (total tokens should be under threshold = 60)
    manager.append(thread_id, "user", "Xin chào")
    manager.append(thread_id, "assistant", "Chào bạn")
    assert manager.compaction_count(thread_id) == 0

    # Append very long messages to exceed threshold
    manager.append(thread_id, "user", "Tôi đang học lập trình AI và xây dựng hệ thống bộ nhớ cho AI Agent để nén các tin nhắn cũ.")
    manager.append(thread_id, "assistant", "Vâng, nén bộ nhớ giúp tiết kiệm ngữ cảnh khi chat.")
    manager.append(thread_id, "user", "Đồng ý, điều này rất quan trọng cho các ứng dụng thực tế.")

    # Compaction should trigger
    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= config.compact_keep_messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced agent remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "user_recall_test"
    thread_a = "thread_a"
    thread_b = "thread_b"

    # 1. Inform name in Thread A
    baseline.reply(user_id, thread_a, "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, thread_a, "Chào bạn, mình tên là DũngCT.")

    # 2. Ask in fresh Thread B
    res_base = baseline.reply(user_id, thread_b, "Bạn có biết mình tên là gì không?")
    res_adv = advanced.reply(user_id, thread_b, "Bạn có biết mình tên là gì không?")

    # Baseline forgets
    assert "DũngCT" not in res_base["response"]
    # Advanced remembers
    assert "DũngCT" in res_adv["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "user_stress_test"
    thread_id = "thread_long"

    # Chat for 10 turns
    for i in range(10):
        baseline.reply(user_id, thread_id, f"Đây là tin nhắn thứ {i} dài dòng để tăng số lượng token ngữ cảnh trong phiên làm việc.")
        advanced.reply(user_id, thread_id, f"Đây là tin nhắn thứ {i} dài dòng để tăng số lượng token ngữ cảnh trong phiên làm việc.")

    # In baseline, prompt context keeps growing (sum of all messages)
    # In advanced, prompt context is capped due to compaction (only User.md + summary + 2 recent messages)
    base_usage = baseline.prompt_token_usage(thread_id)
    adv_usage = advanced.prompt_token_usage(thread_id)

    assert adv_usage < base_usage
    assert advanced.compaction_count(thread_id) > 0
