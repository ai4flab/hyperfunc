"""SQLite/PostgreSQL-backed memory system for CHAT agents.

Provides persistent memory storage with automatic entity extraction
and relevance-based retrieval using full-text search.

Supports:
- SQLite with FTS5 (default, local)
- PostgreSQL with TSVector (multi-tenant, shared)

Uses llm_completion() internally so any LiteLLM-supported model
can be used for memory processing.
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from peewee import (
    SQL,
    CharField,
    DatabaseProxy,
    FloatField,
    IntegerField,
    Model,
    TextField,
)

# Database proxy - will be bound to actual database at runtime
database_proxy = DatabaseProxy()

# Track if we're using PostgreSQL (for FTS differences)
_is_postgres = False


class MemoryType(Enum):
    """Categories of memory entries."""

    FACT = "fact"  # Factual information (e.g., "User's name is John")
    PREFERENCE = "preference"  # User preferences (e.g., "Prefers Python over JS")
    CONTEXT = "context"  # Contextual info (e.g., "Working on a FastAPI project")
    EVENT = "event"  # Events/actions (e.g., "User asked about authentication")
    ENTITY = "entity"  # Named entities (e.g., people, places, projects)


class MemoryModel(Model):
    """Peewee model for memory storage."""

    content = TextField()
    memory_type = CharField(max_length=20, default="fact")
    entities = TextField(default="[]")  # JSON array
    importance = FloatField(default=0.5)
    created_at = FloatField(default=time.time)
    last_accessed = FloatField(default=time.time)
    access_count = IntegerField(default=0)
    metadata = TextField(default="{}")  # JSON object
    # For multi-tenant support
    user_id = CharField(max_length=255, null=True, index=True)

    class Meta:
        database = database_proxy
        table_name = "memories"


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: Optional[int] = None
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    entities: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "entities": json.dumps(self.entities),
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": json.dumps(self.metadata),
            "user_id": self.user_id,
        }

    @classmethod
    def from_model(cls, model: MemoryModel) -> "MemoryEntry":
        """Create from Peewee model instance."""
        return cls(
            id=model.id,
            content=model.content,
            memory_type=MemoryType(model.memory_type),
            entities=json.loads(model.entities),
            importance=model.importance,
            created_at=model.created_at,
            last_accessed=model.last_accessed,
            access_count=model.access_count,
            metadata=json.loads(model.metadata),
            user_id=model.user_id,
        )


# Default extraction prompt
EXTRACTION_PROMPT = """Extract memories from this conversation turn. Return a JSON array of memory objects.

Each memory should have:
- "content": A concise statement of the memory (1-2 sentences)
- "type": One of "fact", "preference", "context", "event", "entity"
- "entities": Array of named entities mentioned (people, places, projects, etc.)
- "importance": Float 0.0-1.0 (how important is this to remember?)

Only extract genuinely useful information worth remembering. Skip trivial exchanges.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

Return ONLY valid JSON array, no other text. If nothing worth remembering, return [].
"""


def _parse_db_url(db_url: str) -> tuple:
    """Parse database URL and return (db_type, connection_params)."""
    if db_url == ":memory:" or db_url.endswith(".db") or db_url.endswith(".sqlite"):
        # SQLite file or in-memory
        return ("sqlite", {"database": db_url})

    if db_url.startswith("sqlite:///"):
        path = db_url[10:]  # Remove sqlite:///
        return ("sqlite", {"database": path if path else ":memory:"})

    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        parsed = urlparse(db_url)
        return (
            "postgres",
            {
                "database": parsed.path[1:] if parsed.path else "postgres",
                "user": parsed.username,
                "password": parsed.password,
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
            },
        )

    raise ValueError(f"Unsupported database URL: {db_url}")


def _create_database(db_url: str):
    """Create appropriate database instance from URL."""
    global _is_postgres

    db_type, params = _parse_db_url(db_url)

    if db_type == "sqlite":
        from playhouse.sqlite_ext import SqliteExtDatabase

        _is_postgres = False
        return SqliteExtDatabase(params["database"], pragmas={"foreign_keys": 1})

    elif db_type == "postgres":
        try:
            from playhouse.pool import PooledPostgresqlDatabase

            _is_postgres = True
            return PooledPostgresqlDatabase(
                params["database"],
                user=params["user"],
                password=params["password"],
                host=params["host"],
                port=params["port"],
                max_connections=8,
                stale_timeout=300,
            )
        except ImportError:
            raise ImportError(
                "PostgreSQL support requires psycopg2. "
                "Install with: pip install hyperfunc[postgres]"
            )

    raise ValueError(f"Unsupported database type: {db_type}")


class Memory:
    """Cross-database memory store with FTS support.

    Supports SQLite (FTS5) and PostgreSQL (TSVector) for full-text search.

    Usage:
        # SQLite (default, local)
        memory = Memory("chat.db")
        memory = Memory(":memory:")  # In-memory
        memory = Memory("sqlite:///path/to/db.sqlite")

        # PostgreSQL (multi-tenant)
        memory = Memory("postgresql://user:pass@localhost/mydb")

        # Store a conversation turn (extracts memories automatically)
        await memory.store_turn(user_msg, assistant_response, user_id="user123")

        # Retrieve relevant memories for a query
        memories = memory.retrieve("What project am I working on?", user_id="user123")

        # Get formatted context for injection
        context = memory.format_context(memories)
    """

    def __init__(
        self,
        db_url: str = ":memory:",
        extraction_model: str = "gpt-4o-mini",
        max_memories_per_turn: int = 5,
        retrieval_limit: int = 10,
    ):
        """
        Initialize memory store.

        Args:
            db_url: Database URL. Supports:
                - ":memory:" or "*.db" for SQLite
                - "sqlite:///path/to/db" for SQLite
                - "postgresql://user:pass@host/db" for PostgreSQL
            extraction_model: LLM model for memory extraction
            max_memories_per_turn: Max memories to extract per conversation turn
            retrieval_limit: Default limit for memory retrieval
        """
        self.db_url = db_url
        self.extraction_model = extraction_model
        self.max_memories_per_turn = max_memories_per_turn
        self.retrieval_limit = retrieval_limit

        # Create and bind database
        self._db = _create_database(db_url)
        database_proxy.initialize(self._db)

        # Create tables
        self._db.connect(reuse_if_open=True)
        self._db.create_tables([MemoryModel])

        # Create FTS index for SQLite
        if not _is_postgres:
            self._setup_sqlite_fts()

    def _setup_sqlite_fts(self) -> None:
        """Set up SQLite FTS5 virtual table."""
        # Check if FTS table exists
        cursor = self._db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        if cursor.fetchone():
            return

        # Create FTS5 virtual table
        self._db.execute_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                entities,
                content='memories',
                content_rowid='id'
            )
            """
        )

        # Create triggers to keep FTS in sync
        self._db.execute_sql(
            """
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, entities)
                VALUES (new.id, new.content, new.entities);
            END
            """
        )

        self._db.execute_sql(
            """
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, entities)
                VALUES ('delete', old.id, old.content, old.entities);
            END
            """
        )

        self._db.execute_sql(
            """
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, entities)
                VALUES ('delete', old.id, old.content, old.entities);
                INSERT INTO memories_fts(rowid, content, entities)
                VALUES (new.id, new.content, new.entities);
            END
            """
        )

    def close(self) -> None:
        """Close database connection."""
        if self._db and not self._db.is_closed():
            self._db.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> int:
        """Add a memory entry directly."""
        model = MemoryModel.create(**entry.to_dict())
        return model.id

    def add_many(self, entries: List[MemoryEntry]) -> List[int]:
        """Add multiple memory entries."""
        ids = []
        with self._db.atomic():
            for entry in entries:
                model = MemoryModel.create(**entry.to_dict())
                ids.append(model.id)
        return ids

    async def store_turn(
        self,
        user_message: str,
        assistant_response: str,
        user_id: Optional[str] = None,
        extract: bool = True,
    ) -> List[MemoryEntry]:
        """
        Store a conversation turn, optionally extracting memories.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            user_id: Optional user ID for multi-tenant storage
            extract: Whether to extract memories using LLM

        Returns:
            List of extracted/stored memory entries
        """
        if not extract:
            # Store as single context entry without extraction
            entry = MemoryEntry(
                content=f"User: {user_message}\nAssistant: {assistant_response}",
                memory_type=MemoryType.CONTEXT,
                importance=0.3,
                user_id=user_id,
            )
            self.add(entry)
            return [entry]

        # Extract memories using LLM
        entries = await self._extract_memories(user_message, assistant_response, user_id)
        self.add_many(entries)
        return entries

    async def _extract_memories(
        self,
        user_message: str,
        assistant_response: str,
        user_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Extract memories from a conversation turn using LLM."""
        from .llm import LITELLM_AVAILABLE, llm_completion

        if not LITELLM_AVAILABLE:
            # Fallback: store as single context entry
            return [
                MemoryEntry(
                    content=f"User: {user_message}\nAssistant: {assistant_response}",
                    memory_type=MemoryType.CONTEXT,
                    importance=0.3,
                    user_id=user_id,
                )
            ]

        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        try:
            response = await llm_completion(
                model=self.extraction_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temp for consistent extraction
                max_tokens=1000,
                rate_limit=True,
            )

            # Parse JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            memories_data = json.loads(content)

            entries: List[MemoryEntry] = []
            for mem in memories_data[: self.max_memories_per_turn]:
                mem_type = mem.get("type", "fact")
                try:
                    memory_type = MemoryType(mem_type)
                except ValueError:
                    memory_type = MemoryType.FACT

                entries.append(
                    MemoryEntry(
                        content=mem.get("content", ""),
                        memory_type=memory_type,
                        entities=mem.get("entities", []),
                        importance=float(mem.get("importance", 0.5)),
                        user_id=user_id,
                    )
                )

            return entries

        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback on parse error
            return [
                MemoryEntry(
                    content=f"User discussed: {user_message[:100]}",
                    memory_type=MemoryType.CONTEXT,
                    importance=0.3,
                    user_id=user_id,
                )
            ]

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.

        Uses full-text search (FTS5 for SQLite, TSVector for PostgreSQL)
        combined with importance scoring.

        Args:
            query: Search query
            user_id: Filter by user ID (for multi-tenant)
            limit: Max memories to return
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold

        Returns:
            List of relevant memory entries, most relevant first
        """
        limit = limit or self.retrieval_limit

        if _is_postgres:
            return self._retrieve_postgres(query, user_id, limit, memory_types, min_importance)
        else:
            return self._retrieve_sqlite(query, user_id, limit, memory_types, min_importance)

    def _retrieve_sqlite(
        self,
        query: str,
        user_id: Optional[str],
        limit: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float,
    ) -> List[MemoryEntry]:
        """SQLite FTS5 retrieval."""
        # Escape special FTS characters
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return self.get_recent(limit, user_id, memory_types, min_importance)

        try:
            # FTS5 query with BM25 ranking
            now = time.time()
            sql = """
                SELECT m.*,
                       bm25(memories_fts) as rank,
                       (m.importance * 0.3 +
                        (1.0 / (1.0 + (? - m.last_accessed) / 86400.0)) * 0.2 +
                        (-bm25(memories_fts)) * 0.5) as score
                FROM memories m
                JOIN memories_fts ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.importance >= ?
            """
            params = [now, safe_query, min_importance]

            if user_id is not None:
                sql += " AND m.user_id = ?"
                params.append(user_id)

            if memory_types:
                placeholders = ",".join("?" * len(memory_types))
                sql += f" AND m.memory_type IN ({placeholders})"
                params.extend(t.value for t in memory_types)

            sql += " ORDER BY score DESC LIMIT ?"
            params.append(limit)

            cursor = self._db.execute_sql(sql, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            # Update access timestamps
            ids = [row[columns.index("id")] for row in rows]
            if ids:
                MemoryModel.update(
                    last_accessed=now, access_count=MemoryModel.access_count + 1
                ).where(MemoryModel.id.in_(ids)).execute()

            # Convert to MemoryEntry
            entries = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                entries.append(
                    MemoryEntry(
                        id=row_dict["id"],
                        content=row_dict["content"],
                        memory_type=MemoryType(row_dict["memory_type"]),
                        entities=json.loads(row_dict["entities"]),
                        importance=row_dict["importance"],
                        created_at=row_dict["created_at"],
                        last_accessed=row_dict["last_accessed"],
                        access_count=row_dict["access_count"],
                        metadata=json.loads(row_dict["metadata"]),
                        user_id=row_dict["user_id"],
                    )
                )
            return entries

        except Exception:
            # Fallback to recent on FTS error
            return self.get_recent(limit, user_id, memory_types, min_importance)

    def _retrieve_postgres(
        self,
        query: str,
        user_id: Optional[str],
        limit: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float,
    ) -> List[MemoryEntry]:
        """PostgreSQL TSVector retrieval."""
        now = time.time()

        # Build base query with ts_rank
        base_query = MemoryModel.select(
            MemoryModel,
            SQL(
                "ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank",
                query,
            ),
        ).where(MemoryModel.importance >= min_importance)

        # Add FTS filter
        base_query = base_query.where(
            SQL(
                "to_tsvector('english', content) @@ plainto_tsquery('english', %s)",
                query,
            )
        )

        if user_id is not None:
            base_query = base_query.where(MemoryModel.user_id == user_id)

        if memory_types:
            type_values = [t.value for t in memory_types]
            base_query = base_query.where(MemoryModel.memory_type.in_(type_values))

        # Order by combined score
        base_query = base_query.order_by(SQL("rank DESC")).limit(limit)

        results = list(base_query)

        # Update access timestamps
        if results:
            ids = [m.id for m in results]
            MemoryModel.update(
                last_accessed=now, access_count=MemoryModel.access_count + 1
            ).where(MemoryModel.id.in_(ids)).execute()

        return [MemoryEntry.from_model(m) for m in results]

    def get_recent(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Get most recent memories."""
        limit = limit or self.retrieval_limit

        query = MemoryModel.select().where(MemoryModel.importance >= min_importance)

        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)

        if memory_types:
            type_values = [t.value for t in memory_types]
            query = query.where(MemoryModel.memory_type.in_(type_values))

        query = query.order_by(MemoryModel.created_at.desc()).limit(limit)

        return [MemoryEntry.from_model(m) for m in query]

    def get_by_entity(
        self,
        entity: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Get memories mentioning a specific entity."""
        limit = limit or self.retrieval_limit

        query = MemoryModel.select().where(
            MemoryModel.entities.contains(f'"{entity}"')
        )

        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)

        query = query.order_by(
            MemoryModel.importance.desc(), MemoryModel.created_at.desc()
        ).limit(limit)

        return [MemoryEntry.from_model(m) for m in query]

    def get_all(self, user_id: Optional[str] = None) -> List[MemoryEntry]:
        """Get all memories (use with caution on large DBs)."""
        query = MemoryModel.select().order_by(MemoryModel.created_at.desc())

        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)

        return [MemoryEntry.from_model(m) for m in query]

    def count(self, user_id: Optional[str] = None) -> int:
        """Get total memory count."""
        query = MemoryModel.select()
        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)
        return query.count()

    # -------------------------------------------------------------------------
    # Context Formatting
    # -------------------------------------------------------------------------

    def format_context(
        self,
        memories: List[MemoryEntry],
        max_chars: int = 2000,
    ) -> str:
        """
        Format memories as context string for injection into prompts.

        Args:
            memories: List of memory entries
            max_chars: Maximum characters in output

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        lines = ["[Relevant memories:]"]
        char_count = len(lines[0])

        for mem in memories:
            line = f"- {mem.content}"
            if char_count + len(line) + 1 > max_chars:
                break
            lines.append(line)
            char_count += len(line) + 1

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def prune(
        self,
        max_age_days: float = 30,
        min_importance: float = 0.2,
        keep_accessed_recently: bool = True,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Prune old, low-importance memories.

        Args:
            max_age_days: Max age in days for low-importance memories
            min_importance: Memories below this importance get pruned
            keep_accessed_recently: Keep if accessed in last 7 days
            user_id: Only prune for specific user

        Returns:
            Number of memories deleted
        """
        cutoff = time.time() - (max_age_days * 86400)
        recent_access = time.time() - (7 * 86400)

        query = MemoryModel.delete().where(
            (MemoryModel.importance < min_importance) & (MemoryModel.created_at < cutoff)
        )

        if keep_accessed_recently:
            query = query.where(MemoryModel.last_accessed < recent_access)

        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)

        return query.execute()

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear all memories (optionally for specific user)."""
        query = MemoryModel.delete()
        if user_id is not None:
            query = query.where(MemoryModel.user_id == user_id)
        query.execute()


# Convenience alias
SQLiteMemory = Memory
