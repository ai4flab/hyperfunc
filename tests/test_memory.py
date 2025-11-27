"""Tests for memory system."""

import pytest

from hyperfunc import Memory, MemoryEntry, MemoryType


@pytest.fixture
def memory():
    """Create a fresh in-memory database for each test."""
    mem = Memory(":memory:")
    yield mem
    mem.close()


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_default_values(self):
        entry = MemoryEntry(content="Test memory")
        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.FACT
        assert entry.entities == []
        assert entry.importance == 0.5
        assert entry.access_count == 0
        assert entry.user_id is None

    def test_custom_values(self):
        entry = MemoryEntry(
            content="User prefers Python",
            memory_type=MemoryType.PREFERENCE,
            entities=["Python", "User"],
            importance=0.8,
            user_id="user123",
        )
        assert entry.memory_type == MemoryType.PREFERENCE
        assert "Python" in entry.entities
        assert entry.importance == 0.8
        assert entry.user_id == "user123"

    def test_to_dict(self):
        entry = MemoryEntry(
            content="Test",
            memory_type=MemoryType.CONTEXT,
            entities=["foo"],
            user_id="user456",
        )
        d = entry.to_dict()
        assert d["content"] == "Test"
        assert d["memory_type"] == "context"
        assert '"foo"' in d["entities"]
        assert d["user_id"] == "user456"


class TestMemoryType:
    """Test MemoryType enum."""

    def test_values(self):
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.CONTEXT.value == "context"
        assert MemoryType.EVENT.value == "event"
        assert MemoryType.ENTITY.value == "entity"


class TestMemoryStorage:
    """Test Memory storage operations."""

    def test_init_in_memory(self, memory):
        assert memory.count() == 0

    def test_add_single(self, memory):
        entry = MemoryEntry(content="Test fact", memory_type=MemoryType.FACT)
        entry_id = memory.add(entry)
        assert entry_id == 1
        assert memory.count() == 1

    def test_add_many(self, memory):
        entries = [
            MemoryEntry(content="Fact 1"),
            MemoryEntry(content="Fact 2"),
            MemoryEntry(content="Fact 3"),
        ]
        ids = memory.add_many(entries)
        assert len(ids) == 3
        assert memory.count() == 3

    def test_get_all(self, memory):
        memory.add(MemoryEntry(content="First"))
        memory.add(MemoryEntry(content="Second"))

        all_memories = memory.get_all()
        assert len(all_memories) == 2
        contents = [m.content for m in all_memories]
        assert "First" in contents
        assert "Second" in contents

    def test_clear(self, memory):
        memory.add(MemoryEntry(content="Test"))
        assert memory.count() == 1
        memory.clear()
        assert memory.count() == 0

    def test_context_manager(self):
        with Memory(":memory:") as mem:
            mem.add(MemoryEntry(content="Test"))
            assert mem.count() == 1


class TestMemoryRetrieval:
    """Test Memory retrieval operations."""

    def test_retrieve_by_fts(self, memory):
        memory.add(MemoryEntry(content="User is building a FastAPI project"))
        memory.add(MemoryEntry(content="Weather is sunny today"))
        memory.add(MemoryEntry(content="FastAPI uses Python"))

        results = memory.retrieve("FastAPI")
        assert len(results) >= 1
        # Should find FastAPI-related memories
        contents = [r.content for r in results]
        assert any("FastAPI" in c for c in contents)

    def test_retrieve_updates_access(self, memory):
        memory.add(MemoryEntry(content="Important fact"))

        # First retrieve
        results = memory.retrieve("Important")
        assert len(results) == 1

        # Access count is updated in DB after retrieval
        # A second retrieval should show the updated count
        results2 = memory.retrieve("Important")
        assert len(results2) == 1
        assert results2[0].access_count == 1  # Now shows the count from first retrieval

    def test_get_recent(self, memory):
        memory.add(MemoryEntry(content="Old memory", importance=0.5))
        memory.add(MemoryEntry(content="New memory", importance=0.5))

        recent = memory.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].content == "New memory"

    def test_get_by_entity(self, memory):
        memory.add(
            MemoryEntry(
                content="John works at Acme",
                entities=["John", "Acme"],
            )
        )
        memory.add(
            MemoryEntry(
                content="Weather is nice",
                entities=[],
            )
        )

        results = memory.get_by_entity("John")
        assert len(results) == 1
        assert "John" in results[0].entities

    def test_filter_by_importance(self, memory):
        memory.add(MemoryEntry(content="Low importance", importance=0.1))
        memory.add(MemoryEntry(content="High importance", importance=0.9))

        results = memory.get_recent(min_importance=0.5)
        assert len(results) == 1
        assert results[0].importance == 0.9

    def test_filter_by_type(self, memory):
        memory.add(MemoryEntry(content="A fact", memory_type=MemoryType.FACT))
        memory.add(MemoryEntry(content="A preference", memory_type=MemoryType.PREFERENCE))

        results = memory.get_recent(memory_types=[MemoryType.FACT])
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.FACT


class TestMultiTenant:
    """Test multi-tenant (user_id) support."""

    def test_filter_by_user_id(self, memory):
        memory.add(MemoryEntry(content="User A memory", user_id="user_a"))
        memory.add(MemoryEntry(content="User B memory", user_id="user_b"))
        memory.add(MemoryEntry(content="Shared memory", user_id=None))

        user_a_memories = memory.get_all(user_id="user_a")
        assert len(user_a_memories) == 1
        assert user_a_memories[0].content == "User A memory"

        user_b_memories = memory.get_all(user_id="user_b")
        assert len(user_b_memories) == 1
        assert user_b_memories[0].content == "User B memory"

    def test_count_by_user_id(self, memory):
        memory.add(MemoryEntry(content="User A - 1", user_id="user_a"))
        memory.add(MemoryEntry(content="User A - 2", user_id="user_a"))
        memory.add(MemoryEntry(content="User B - 1", user_id="user_b"))

        assert memory.count(user_id="user_a") == 2
        assert memory.count(user_id="user_b") == 1
        assert memory.count() == 3

    def test_clear_by_user_id(self, memory):
        memory.add(MemoryEntry(content="User A memory", user_id="user_a"))
        memory.add(MemoryEntry(content="User B memory", user_id="user_b"))

        memory.clear(user_id="user_a")

        assert memory.count(user_id="user_a") == 0
        assert memory.count(user_id="user_b") == 1

    def test_retrieve_by_user_id(self, memory):
        memory.add(MemoryEntry(content="FastAPI for user A", user_id="user_a"))
        memory.add(MemoryEntry(content="FastAPI for user B", user_id="user_b"))

        results = memory.retrieve("FastAPI", user_id="user_a")
        assert len(results) == 1
        assert results[0].user_id == "user_a"


class TestMemoryFormatting:
    """Test Memory context formatting."""

    def test_format_empty(self, memory):
        context = memory.format_context([])
        assert context == ""

    def test_format_single(self, memory):
        entries = [MemoryEntry(content="User likes Python")]
        context = memory.format_context(entries)
        assert "[Relevant memories:]" in context
        assert "User likes Python" in context

    def test_format_multiple(self, memory):
        entries = [
            MemoryEntry(content="Fact 1"),
            MemoryEntry(content="Fact 2"),
        ]
        context = memory.format_context(entries)
        assert "Fact 1" in context
        assert "Fact 2" in context

    def test_format_respects_max_chars(self, memory):
        entries = [
            MemoryEntry(content="A" * 100),
            MemoryEntry(content="B" * 100),
            MemoryEntry(content="C" * 100),
        ]
        context = memory.format_context(entries, max_chars=150)
        # Should truncate to fit within limit
        assert len(context) <= 200  # Some overhead for formatting


class TestMemoryPruning:
    """Test Memory maintenance operations."""

    def test_prune_old_low_importance(self, memory):
        import time

        # Add entry with old timestamp
        old_entry = MemoryEntry(
            content="Old low importance",
            importance=0.1,
            created_at=time.time() - (60 * 86400),  # 60 days ago
            last_accessed=time.time() - (60 * 86400),
        )
        memory.add(old_entry)

        # Add recent entry
        memory.add(MemoryEntry(content="Recent", importance=0.1))

        assert memory.count() == 2

        # Prune old, low-importance memories
        deleted = memory.prune(max_age_days=30, min_importance=0.2)
        assert deleted == 1
        assert memory.count() == 1

    def test_prune_keeps_high_importance(self, memory):
        import time

        # Add old but high importance entry
        entry = MemoryEntry(
            content="Important old memory",
            importance=0.9,
            created_at=time.time() - (60 * 86400),
        )
        memory.add(entry)

        deleted = memory.prune(max_age_days=30, min_importance=0.5)
        assert deleted == 0
        assert memory.count() == 1


class TestDatabaseURLParsing:
    """Test database URL parsing."""

    def test_sqlite_memory(self):
        mem = Memory(":memory:")
        assert mem.count() == 0
        mem.close()

    def test_sqlite_file_extension(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            mem = Memory(db_path)
            mem.add(MemoryEntry(content="Test"))
            assert mem.count() == 1
            mem.close()

    def test_sqlite_url_format(self):
        mem = Memory("sqlite:///:memory:")
        assert mem.count() == 0
        mem.close()


@pytest.mark.asyncio
class TestMemoryStoreTurn:
    """Test async store_turn method."""

    async def test_store_turn_no_extract(self, memory):
        entries = await memory.store_turn(
            user_message="Hello!",
            assistant_response="Hi there!",
            extract=False,
        )

        assert len(entries) == 1
        assert entries[0].memory_type == MemoryType.CONTEXT
        assert "Hello!" in entries[0].content
        assert "Hi there!" in entries[0].content

    async def test_store_turn_with_user_id(self, memory):
        entries = await memory.store_turn(
            user_message="Hello!",
            assistant_response="Hi there!",
            user_id="user123",
            extract=False,
        )

        assert len(entries) == 1
        assert entries[0].user_id == "user123"
