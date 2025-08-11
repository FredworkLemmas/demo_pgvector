import attrs
import psycopg2

from .interfaces import PostgresqlConnectionProvider


@attrs.define
class PgvectorDatabaseConnection(PostgresqlConnectionProvider):
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
            password=self.password
        )
        return connection

    @classmethod
    def from_settings_provider(cls, settings: dict):
        return cls(
            host=settings['database']['host'],
            port=settings['database']['port'],
            database=settings['database']['database'],
            user=settings['database']['user'],
            password=settings['database']['password']
        )
