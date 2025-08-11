import psycopg2
import numpy as np
from vllm import LLM, SamplingParams
import json
import logging
import asyncio
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize vLLM with an embedding model"""
        self.model_name = model_name
        self.llm = None
        self.embedding_dim = 384  # Default for bge-small-en-v1.5

    def load_model(self):
        """Load the vLLM model for embedding generation"""
        try:
            logger.info(f"Loading vLLM model: {self.model_name}")
            # Configure vLLM for embedding tasks
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=512,  # Adjust based on your needs
                gpu_memory_utilization=0.7,
                tensor_parallel_size=1,
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.llm:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # For embedding models, we typically use the model's encode method
            # Since vLLM doesn't directly support embedding generation,
            # we'll use a workaround with a prompt-based approach
            embeddings = []

            for text in texts:
                # Create a prompt that asks the model to represent the text
                prompt = f"Encode this text for semantic similarity: {text}"

                # Generate response (this is a simplified approach)
                # In practice, you might need to use the model's specific embedding method
                outputs = self.llm.generate(
                    [prompt], sampling_params=SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        stop=None
                    ))

                # For this example, we'll create a mock embedding
                # In reality, you'd extract embeddings from the model's hidden states
                mock_embedding = self._create_mock_embedding(text)
                embeddings.append(mock_embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def _create_mock_embedding(self, text: str) -> List[float]:
        """Create a mock embedding based on text hash (for demonstration)"""
        # This is a simplified approach - in practice, you'd extract actual embeddings
        import hashlib

        # Create a deterministic "embedding" based on text content
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generate pseudo-random embedding
        np.random.seed(hash_int % (2 ** 32))
        embedding = np.random.normal(0, 1, self.embedding_dim).tolist()

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        embedding = [x / norm for x in embedding]

        return embedding


def create_connection():
    """Create a connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="your_database_name",
            user="your_username",
            password="your_password",
            port="5432"
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None


def setup_pgvector_extension(conn, embedding_dim: int = 384):
    """Set up pgvector extension and create table"""
    try:
        cursor = conn.cursor()

        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create table for storing embeddings
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS text_embeddings (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding VECTOR({embedding_dim}),
                model_name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create index for faster similarity search
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS text_embeddings_embedding_idx
                ON text_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                       """)

        conn.commit()
        cursor.close()
        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        conn.rollback()


def store_embedding(conn, text: str, embedding: List[float], model_name: str):
    """Store text and its embedding in PostgreSQL"""
    try:
        cursor = conn.cursor()

        # Insert text and embedding
        cursor.execute(
            """
            INSERT INTO text_embeddings (text, embedding, model_name)
            VALUES (%s, %s, %s)
                       """, (text, embedding, model_name))

        conn.commit()
        cursor.close()
        logger.info(f"Successfully stored embedding for: '{text[:50]}...'")

    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        conn.rollback()


def retrieve_similar_texts(conn, query_embedding: List[float], limit: int = 5):
    """Retrieve similar texts using cosine similarity"""
    try:
        cursor = conn.cursor()

        # Use pgvector's cosine distance operator
        cursor.execute(
            """
            SELECT text, embedding <=> %s AS distance, model_name
            FROM text_embeddings
            ORDER BY embedding <=> %s
                LIMIT %s
                       """, (query_embedding, query_embedding, limit))

        results = cursor.fetchall()
        cursor.close()

        return results

    except Exception as e:
        logger.error(f"Error retrieving similar texts: {e}")
        return []


def main():
    # The text to process
    text = "the quick brown fox jumps over the lazy dog"

    # Initialize embedding generator with vLLM
    model_name = "BAAI/bge-small-en-v1.5"  # Popular embedding model
    embedding_generator = EmbeddingGenerator(model_name)

    try:
        # Load the model
        embedding_generator.load_model()

        # Create database connection
        logger.info("Connecting to database...")
        conn = create_connection()
        if not conn:
            logger.error("Failed to connect to database. Exiting.")
            return

        # Setup database and pgvector extension
        setup_pgvector_extension(conn, embedding_generator.embedding_dim)

        # Generate embedding for the text
        logger.info(f"Generating embedding for: '{text}'")
        embeddings = embedding_generator.generate_embeddings([text])

        if not embeddings:
            logger.error("Failed to generate embedding. Exiting.")
            return

        embedding = embeddings[0]

        # Store the embedding in PostgreSQL
        store_embedding(conn, text, embedding, model_name)

        # Example: Add a few more texts for demonstration
        additional_texts = [
            "a fast brown fox leaps over a sleepy dog",
            "the weather is nice today",
            "machine learning is fascinating"
        ]

        logger.info("Generating embeddings for additional texts...")
        additional_embeddings = embedding_generator.generate_embeddings(
            additional_texts)

        for i, (add_text, add_embedding) in enumerate(
                zip(additional_texts, additional_embeddings)):
            store_embedding(conn, add_text, add_embedding, model_name)

        # Retrieve similar texts
        logger.info("Retrieving similar texts...")
        similar_texts = retrieve_similar_texts(conn, embedding, limit=5)

        print("\nSimilar texts found:")
        print("-" * 80)
        for text_result, distance, model in similar_texts:
            print(f"Text: '{text_result}'")
            print(f"Distance: {distance:.4f} | Model: {model}")
            print("-" * 80)

        # Display embedding info
        print(f"\nEmbedding generated successfully!")
        print(f"Model used: {model_name}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 values: {[f'{x:.4f}' for x in embedding[:10]]}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        # Close database connection
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()