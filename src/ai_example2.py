import psycopg2
import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class vLLMEmbeddingGenerator:
    def __init__(
            self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: list) -> list:
        """Generate embeddings for a list of texts"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            embeddings = []

            with torch.no_grad():
                for text in texts:
                    # Tokenize text
                    encoded_input = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)

                    # Generate embeddings
                    model_output = self.model(**encoded_input)

                    # Apply mean pooling
                    sentence_embedding = self.mean_pooling(
                        model_output, encoded_input['attention_mask'])

                    # Normalize embedding
                    sentence_embedding = F.normalize(
                        sentence_embedding, p=2, dim=1)

                    # Convert to list
                    embedding = sentence_embedding.cpu().numpy().flatten().tolist()
                    embeddings.append(embedding)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []


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
            CREATE TABLE IF NOT EXISTS vllm_text_embeddings (
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
            CREATE INDEX IF NOT EXISTS vllm_embeddings_idx
                ON vllm_text_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                       """)

        conn.commit()
        cursor.close()
        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        conn.rollback()


def store_embedding(conn, text: str, embedding: list, model_name: str):
    """Store text and its embedding in PostgreSQL"""
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO vllm_text_embeddings (text, embedding, model_name)
            VALUES (%s, %s, %s)
                       """, (text, embedding, model_name))

        conn.commit()
        cursor.close()
        logger.info(f"Successfully stored embedding for: '{text[:50]}...'")

    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        conn.rollback()


def retrieve_similar_texts(conn, query_embedding: list, limit: int = 5):
    """Retrieve similar texts using cosine similarity"""
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT text, embedding <=> %s AS distance, model_name
            FROM vllm_text_embeddings
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

    # Initialize embedding generator
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_generator = vLLMEmbeddingGenerator(model_name)

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
        setup_pgvector_extension(
            conn, 384)  # all-MiniLM-L6-v2 has 384 dimensions

        # Generate embedding for the text
        logger.info(f"Generating embedding for: '{text}'")
        embeddings = embedding_generator.generate_embeddings([text])

        if not embeddings:
            logger.error("Failed to generate embedding. Exiting.")
            return

        embedding = embeddings[0]

        # Store the embedding in PostgreSQL
        store_embedding(conn, text, embedding, model_name)

        # Add some similar and different texts for demonstration
        additional_texts = [
            "a quick brown fox jumps over the lazy dog",  # Very similar
            "the fast brown fox leaps over a sleepy dog",  # Similar
            "dogs and cats are pets",  # Somewhat related
            "the weather is sunny today",  # Different topic
        ]

        logger.info("Generating embeddings for additional texts...")
        additional_embeddings = embedding_generator.generate_embeddings(
            additional_texts)

        for add_text, add_embedding in zip(
                additional_texts, additional_embeddings):
            store_embedding(conn, add_text, add_embedding, model_name)

        # Retrieve similar texts
        logger.info("Retrieving similar texts...")
        similar_texts = retrieve_similar_texts(conn, embedding, limit=6)

        print(f"\nQuery text: '{text}'")
        print("\nSimilar texts found (ordered by similarity):")
        print("=" * 80)
        for i, (text_result, distance, model) in enumerate(similar_texts, 1):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"{i}. Text: '{text_result}'")
            print(f"   Similarity: {similarity:.4f} | Distance: {distance:.4f}")
            print(f"   Model: {model}")
            print("-" * 80)

        # Display embedding info
        print(f"\nEmbedding Information:")
        print(f"Model used: {model_name}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"Device used: {embedding_generator.device}")
        print(
            f"First 10 embedding values: {[f'{x:.4f}' for x in embedding[:10]]}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        # Close database connection
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()