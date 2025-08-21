from typing import Optional

import attrs
import psycopg2

from .documents import SourceDocument, TEXT_TYPE__FICTION, TEXT_TYPE__NONFICTION
from .interfaces import PostgresqlConnectionProvider, SettingsProvider


@attrs.define
class PgvectorDatabaseConnectionProvider(PostgresqlConnectionProvider):
    host: str
    port: int
    database: str
    user: str
    password: str

    def get_connection(self):
        connection = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        return connection

    @classmethod
    def from_settings_provider(cls, settings_provider: SettingsProvider):
        settings = settings_provider.get_settings()
        return cls(
            host=settings['database']['host'],
            port=settings['database']['port'],
            database=settings['database']['database'],
            user=settings['database']['user'],
            password=settings['database']['password'],
        )


class SimpleVectorDatabase:
    def __init__(self, connection_provider: PostgresqlConnectionProvider):
        self.connection_provider = connection_provider

    @classmethod
    def from_settings_provider(cls, settings_provider: SettingsProvider):
        connection_provider = (
            PgvectorDatabaseConnectionProvider.from_settings_provider(
                settings_provider
            )
        )
        return cls(connection_provider=connection_provider)

    def create_or_lookup_model_id(self, model_name: str) -> int:
        # get connection
        conn = self.connection_provider.get_connection()
        cursor = conn.cursor()

        try:
            # First, try to find existing model
            cursor.execute(
                'SELECT id FROM models WHERE name = %s', (model_name,)
            )
            result = cursor.fetchone()

            if result:
                # Model exists, return its id
                return result[0]
            else:
                # Model doesn't exist, create it
                cursor.execute(
                    'INSERT INTO models (name, embedding_dim) VALUES (%s, %s) RETURNING id',
                    (model_name, self.embedding_dim),
                )
                model_id = cursor.fetchone()[0]

                conn.commit()
                return model_id

        finally:
            cursor.close()

    def create_or_lookup_source(self, doc: SourceDocument) -> int:
        """
        If a source row with the same identifying data already exists, return its id.
        Otherwise, insert a new row (same behavior as insert_into_sources) and return the new id.
        Identifying data uses the unique key: (model_id, author, title, year).
        """

        # Map text_type to source_type (simple heuristic)
        def _map_source_type(meta: dict) -> int:
            text_type = (meta.get('text_type') or '').strip().lower()
            fiction_indicators = {
                'short story',
                'novel',
                'novella',
                'poem',
                'fiction',
                'science fiction',
                'sci-fi',
                'sf',
                'fantasy',
                'fable',
                'drama',
                'play',
            }
            if text_type in fiction_indicators:
                return TEXT_TYPE__FICTION
            # If genre clearly implies fiction, treat it as fiction.
            genre = (meta.get('genre') or '').strip().lower()
            if genre in {
                'science fiction',
                'fantasy',
                'horror',
                'mystery',
                'thriller',
            }:
                return TEXT_TYPE__FICTION
            return TEXT_TYPE__NONFICTION

        def _extract_year(meta: dict) -> Optional[int]:
            # Expecting 'publication_date' possibly as int, 'YYYY', or ISO date.
            pub = meta.get('publication_date')
            if pub is None:
                return None
            try:
                return int(str(pub)[:4])
            except Exception:
                return None

        meta = doc.metadata or {}

        author = meta.get('author')
        title = meta.get('title')
        url = meta.get('url')
        genre = meta.get('genre')
        subgenre = meta.get('subgenre')
        source_type = _map_source_type(meta)
        year = _extract_year(meta)

        conn = self.connection_provider.get_connection()
        conn.autocommit = False

        # First, try to find an existing source by the unique key (null-safe comparisons).
        select_sql = """
            SELECT id
            FROM sources
            WHERE model_id = %s
              AND author IS NOT DISTINCT FROM %s
              AND title IS NOT DISTINCT FROM %s
              AND year IS NOT DISTINCT FROM %s
        """

        try:
            with conn.cursor() as cur:
                cur.execute(select_sql, (doc.model_id, author, title, year))
                row = cur.fetchone()
                if row:
                    conn.commit()
                    return row[0]

                # Not found, attempt to insert (same as insert_into_sources)
                try:
                    cur.execute(
                        """
                        INSERT INTO sources (author, title, source_type, url, genre,
                                             subgenre, year, model_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            author,
                            title,
                            source_type,
                            url,
                            genre,
                            subgenre,
                            year,
                            doc.model_id,
                        ),
                    )
                    new_id = cur.fetchone()[0]
                    conn.commit()
                    return new_id
                except psycopg2.IntegrityError:
                    # Another concurrent transaction inserted the same unique row.
                    conn.rollback()
                    with conn.cursor() as cur2:
                        cur2.execute(
                            select_sql, (doc.model_id, author, title, year)
                        )
                        row2 = cur2.fetchone()
                        if row2:
                            conn.commit()
                            return row2[0]
                        # If still not found, re-raise to surface the issue.
                        raise
        finally:
            # Ensure connection is not left in a bad transactional state.
            try:
                if (
                    conn.closed == 0
                    and conn.get_transaction_status()
                    != psycopg2.extensions.TRANSACTION_STATUS_IDLE
                ):
                    conn.rollback()
            except Exception:
                pass
