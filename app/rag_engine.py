



"""

import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
class CareerCompassWeaviate:
    def __init__(self):
        self.client = None
        self.vectorstore = None
        # Initialize OpenAI client only when needed
        self._llm_client = None
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._llm_client
    
    # --- 1️⃣ Connect to Weaviate Cloud ---
    def _initialize_weaviate_client(self):
        Initialize connection to Weaviate Cloud
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            print(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            print("📂 Current working directory:", os.getcwd())
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(api_key)
            )

            if self.client.is_ready():
                print("✅ Connected to Weaviate Cloud")
                return True
            else:
                print("❌ Weaviate not ready")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    # --- 2️⃣ Schema Management ---
    def _check_and_create_schema(self):
        Ensure the schema (CareerKnowledge) exists
        try:
            class_name = "CareerKnowledge"
            schema = self.client.collections.list_all()

            if class_name not in schema:
                print("📋 Creating Weaviate schema...")
                self.client.collections.create(
                    name=class_name,
                    properties=[
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="is_augmented", data_type=DataType.BOOL),
                        Property(name="source", data_type=DataType.TEXT),
                    ]
                )
                print("✅ Schema created")
            else:
                print("✅ Schema already exists")

            return True
        except Exception as e:
            print(f"❌ Schema creation error: {e}")
            return False

    # --- 3️⃣ Initialize Data + Embeddings ---
    def initialize_system(self, data_path):
        
        print("🚀 Initializing Career Compass RAG...")

        if not self._initialize_weaviate_client():
            print("❌ Weaviate client init failed")
            return False

        if not self._check_and_create_schema():
            print("❌ Schema creation failed")
            return False

        try:
            df = pd.read_csv(data_path)
            print(f"📄 Loaded {len(df)} rows from CSV")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False

        # Chunk text
        print("✂️ Splitting text into token chunks...")
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

        documents = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row["answer"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "question": row["question"],
                        "answer": row["answer"],
                        "is_augmented": False,
                        "source": "career_compass_dataset"
                    }
                )
                documents.append(doc)

        print(f"📝 Prepared {len(documents)} chunks")

        # Embeddings
        print("🧠 Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Vector store
        print("💾 Creating vector store in Weaviate...")
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name="CareerKnowledge",
            text_key="answer",
            embedding=embedding_model,
            attributes=["question", "answer", "is_augmented", "source"]
        )

        # Add documents
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.vectorstore.add_documents(documents[i:i + batch_size])
            print(f"📤 Added {min(i + batch_size, len(documents))}/{len(documents)}")

        print("✅ All documents added successfully")
        print("🎉 Career Compass RAG is ready!")
        return True

    # --- 4️⃣ Ask a Question ---
    def ask_question(self, question):
        Retrieve + Generate an answer using RAG
        try:
            if not self.vectorstore:
                return {"answer": "System not initialized.", "confidence": "Error"}

            results = self.vectorstore.similarity_search(query=question, k=5)

            if not results:
                return {"answer": "I don't have enough information.", "confidence": "Low"}

            context = "\n".join([doc.page_content for doc in results])

            prompt = f
            You are Career Compass, a career guidance assistant.

            Use the context below to answer the question.

            Context:
            {context}

            Question: {question}
            Answer:
            

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = response.choices[0].message.content.strip()
            return {
                "answer": final_answer,
                "retrieved_chunks": len(results),
                "confidence": "High"
            }

        except Exception as e:
            return {"answer": f"Error: {e}", "confidence": "Error"}

    # --- 5️⃣ Cleanup ---
    def close_connection(self):
        Close connection to Weaviate
        if self.client:
            self.client.close()
            print("🔌 Connection closed.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")

    print("📁 Looking for dataset at:", csv_path)

    system = CareerCompassWeaviate()
    system.initialize_system(csv_path)
    response = system.ask_question("What skills are important for AI engineers?")
    print("💡 Answer:", response["answer"])
    system.close_connection()
"""

import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv() 
llm_client = OpenAI()

class CareerCompassWeaviate:
    def __init__(self):
        self.client = None
        self.vectorstore = None
        self.llm_client = None
    def llm_client(self):
        if self._llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self._llm_client = OpenAI(api_key=api_key)
        return self._llm_client
    # --- 1️⃣ Connect to Weaviate Cloud ---
    def _initialize_weaviate_client(self):
        """Initialize connection to Weaviate Cloud"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")
            if not cluster_url:
                raise ValueError("WEAVIATE_CLOUD_URL environment variable is not set")
            if not api_key:
                raise ValueError("WEAVIATE_API_KEY environment variable is not set")
            print(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(api_key)
            )

            if self.client.is_ready():
                print("✅ Connected to Weaviate Cloud")
                return True
            else:
                print("❌ Weaviate not ready")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    # --- 2️⃣ Schema Management ---
    def _check_and_create_schema(self):
        """Ensure the schema (CareerKnowledge) exists"""
        try:
            class_name = "CareerKnowledge"
            schema = self.client.collections.list_all()

            if class_name not in schema:
                print("📋 Creating Weaviate schema...")
                self.client.collections.create(
                    name=class_name,
                    properties=[
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="is_augmented", data_type=DataType.BOOL),
                        Property(name="source", data_type=DataType.TEXT),
                    ]
                )
                print("✅ Schema created")
            else:
                print("✅ Schema already exists")

            return True
        except Exception as e:
            print(f"❌ Schema creation error: {e}")
            return False

    # --- 3️⃣ Initialize Data + Embeddings ---
    def initialize_system(self, data_path):
        """Initialize the complete RAG system"""
        print("🚀 Initializing Career Compass RAG...")

        if not self._initialize_weaviate_client():
            print("❌ Weaviate client init failed")
            return False

        if not self._check_and_create_schema():
            print("❌ Schema creation failed")
            return False

        try:
            df = pd.read_csv(data_path)
            print(f"📄 Loaded {len(df)} rows from CSV")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False

        # Chunk text
        print("✂️ Splitting text into token chunks...")
        text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=50)

        documents = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row["answer"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "question": row["question"],
                        "answer": row["answer"],
                        "is_augmented": False,
                        "source": "career_compass_dataset"
                    }
                )
                documents.append(doc)

        print(f"📝 Prepared {len(documents)} chunks")

        # Embeddings
        print("🧠 Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Vector store
        print("💾 Creating vector store in Weaviate...")
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name="CareerKnowledge",
            text_key="answer",
            embedding=embedding_model,
            attributes=["question", "answer", "is_augmented", "source"]
        )

        # Add documents
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.vectorstore.add_documents(documents[i:i + batch_size])
            print(f"📤 Added {min(i + batch_size, len(documents))}/{len(documents)}")

        print("✅ All documents added successfully")
        print("🎉 Career Compass RAG is ready!")
        return True

    # --- 4️⃣ Ask a Question ---
    def ask_question(self, question):
        """Retrieve + Generate an answer using RAG"""
        try:
            if not self.vectorstore:
                return {"answer": "System not initialized.", "confidence": "Error"}

            results = self.vectorstore.similarity_search(query=question, k=5)

            if not results:
                return {"answer": "I don't have enough information.", "confidence": "Low"}

            context = "\n".join([doc.page_content for doc in results])

            prompt = f"""
            You are Career Compass, a career guidance assistant.

            Use the context below to answer the question.

            Context:
            {context}

            Question: {question}
            Answer:
            """

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            final_answer = response.choices[0].message.content.strip()
            return {
                "answer": final_answer,
                "retrieved_chunks": len(results),
                "confidence": "High"
            }

        except Exception as e:
            return {"answer": f"Error: {e}", "confidence": "Error"}

    # --- 5️⃣ Cleanup ---
    def close_connection(self):
        """Close connection to Weaviate"""
        if self.client:
            self.client.close()
            print("🔌 Connection closed.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")

    print("📁 Looking for dataset at:", csv_path)

    system = CareerCompassWeaviate()
    system.initialize_system(csv_path)
    response = system.ask_question("What skills are important for AI engineers?")
    print("💡 Answer:", response["answer"])
    system.close_connection()


