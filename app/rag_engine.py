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
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class CareerCompassWeaviate:
    def __init__(self):
        self.client = None
        self.vectorstore = None
        self._llm_client = None
        self._is_initialized = False
        self._embedding_model = None
        logger.info("CareerCompassWeaviate initialized")

    @property
    def llm_client(self):
        """Lazy initialization of OpenAI client"""
        if self._llm_client is None:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found in environment variables")
                    return None
                self._llm_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        return self._llm_client

    # --- 1️⃣ Connect to Weaviate Cloud ---
    def _initialize_weaviate_client(self):
        """Initialize connection to Weaviate Cloud"""
        try:
            cluster_url = os.getenv("WEAVIATE_CLOUD_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")
            
            if not cluster_url:
                logger.error("WEAVIATE_CLOUD_URL environment variable is not set")
                return False
            if not api_key:
                logger.error("WEAVIATE_API_KEY environment variable is not set")
                return False

            logger.info(f"🔗 Connecting to Weaviate Cloud: {cluster_url}")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(api_key)
            )

            if self.client.is_ready():
                logger.info("✅ Connected to Weaviate Cloud")
                return True
            else:
                logger.error("❌ Weaviate not ready")
                return False
                
        except Exception as e:
            logger.error(f"❌ Weaviate connection error: {e}")
            return False

    # --- 2️⃣ Schema Management ---
    def _check_and_create_schema(self):
        """Ensure the schema (CareerKnowledge) exists"""
        try:
            class_name = "CareerKnowledge"
            collections = self.client.collections.list_all()
            collection_names = [col.name for col in collections]

            if class_name not in collection_names:
                logger.info("📋 Creating Weaviate schema...")
                self.client.collections.create(
                    name=class_name,
                    properties=[
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="is_augmented", data_type=DataType.BOOL),
                        Property(name="source", data_type=DataType.TEXT),
                    ]
                )
                logger.info("✅ Schema created")
            else:
                logger.info("✅ Schema already exists")

            return True
        except Exception as e:
            logger.error(f"❌ Schema creation error: {e}")
            return False

    # --- 3️⃣ Lazy Initialization ---
    def ensure_initialized(self, data_path=None):
        """Lazy initialization - only initialize when needed"""
        if not self._is_initialized:
            if data_path and os.path.exists(data_path):
                return self.initialize_system(data_path)
            else:
                logger.warning("⚠️ No data path provided, initializing with empty vector store")
                return self._initialize_minimal()
        return True

    def _initialize_minimal(self):
        """Initialize minimal components without data loading"""
        try:
            if not self._initialize_weaviate_client():
                return False
            if not self._check_and_create_schema():
                return False
            
            # Initialize embedding model
            self._embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize empty vector store
            self.vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name="CareerKnowledge",
                text_key="answer",
                embedding=self._embedding_model,
                attributes=["question", "answer", "is_augmented", "source"]
            )
            
            self._is_initialized = True
            logger.info("✅ Minimal RAG system initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Minimal initialization failed: {e}")
            return False

    # --- 4️⃣ Initialize Data + Embeddings ---
    def initialize_system(self, data_path):
        """Initialize the complete RAG system with optimizations for production"""
        logger.info("🚀 Initializing Career Compass RAG...")

        if not self._initialize_weaviate_client():
            logger.error("❌ Weaviate client init failed")
            return False

        if not self._check_and_create_schema():
            logger.error("❌ Schema creation failed")
            return False

        # Load and process data
        try:
            if not os.path.exists(data_path):
                logger.error(f"❌ Data file not found: {data_path}")
                return self._initialize_minimal()
                
            logger.info(f"📄 Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"📄 Loaded {len(df)} rows from CSV")
            
            # Sample data for faster initialization in production
            if len(df) > 200:
                logger.info("🔍 Sampling data for faster initialization...")
                df = df.sample(n=min(200, len(df)), random_state=42)
                logger.info(f"📊 Using {len(df)} sampled rows")
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return self._initialize_minimal()

        # Chunk text with error handling
        logger.info("✂️ Splitting text into token chunks...")
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)

        documents = []
        successful_rows = 0
        
        for idx, row in df.iterrows():
            try:
                if pd.notna(row.get("answer")) and str(row.get("answer", "")).strip():
                    answer_text = str(row.get("answer", ""))
                    chunks = text_splitter.split_text(answer_text)
                    
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "question": row.get("question", ""),
                                "answer": row.get("answer", ""),
                                "is_augmented": False,
                                "source": "career_compass_dataset"
                            }
                        )
                        documents.append(doc)
                    successful_rows += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue

        logger.info(f"📝 Prepared {len(documents)} chunks from {successful_rows} rows")

        # Initialize embedding model
        logger.info("🧠 Initializing embedding model...")
        try:
            self._embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"❌ Embedding model failed: {e}")
            return False

        # Initialize vector store
        logger.info("💾 Creating vector store in Weaviate...")
        try:
            self.vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name="CareerKnowledge",
                text_key="answer",
                embedding=self._embedding_model,
                attributes=["question", "answer", "is_augmented", "source"]
            )
        except Exception as e:
            logger.error(f"❌ Vector store creation failed: {e}")
            return False

        # Add documents with progress and error handling
        batch_size = 50
        successful_docs = 0
        
        if documents:
            for i in range(0, len(documents), batch_size):
                try:
                    batch = documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                    successful_docs += len(batch)
                    logger.info(f"📤 Added {successful_docs}/{len(documents)} documents")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to add batch {i}: {e}")
                    continue
        else:
            logger.warning("No documents to add to vector store")

        logger.info(f"✅ {successful_docs}/{len(documents)} documents added successfully")
        
        if successful_docs > 0:
            self._is_initialized = True
            logger.info("🎉 Career Compass RAG is ready!")
            return True
        else:
            logger.error("❌ No documents were added successfully")
            return self._initialize_minimal()

    # --- 5️⃣ Ask a Question ---
    def ask_question(self, question):
        """Retrieve + Generate an answer using RAG with better error handling"""
        try:
            # Ensure system is initialized
            if not self._is_initialized:
                return {
                    "answer": "System is still initializing. Please try again in a moment.", 
                    "confidence": "Low",
                    "retrieved_chunks": 0
                }

            if not self.vectorstore:
                return {
                    "answer": "Knowledge base is not available at the moment.", 
                    "confidence": "Error",
                    "retrieved_chunks": 0
                }

            # Validate question
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.", 
                    "confidence": "Low",
                    "retrieved_chunks": 0
                }

            cleaned_question = question.strip()
            logger.info(f"🔍 Searching for: {cleaned_question}")

            # Retrieve context with error handling
            try:
                results = self.vectorstore.similarity_search(
                    query=cleaned_question, 
                    k=3
                )
                logger.info(f"📚 Retrieved {len(results)} relevant chunks")
            except Exception as e:
                logger.error(f"❌ Vector search error: {e}")
                return {
                    "answer": "Error searching knowledge base. Please try again.", 
                    "confidence": "Error",
                    "retrieved_chunks": 0
                }

            if not results:
                return {
                    "answer": "I don't have enough information about this topic in my knowledge base. Please try asking about career paths, skills, education requirements, or job market trends.", 
                    "confidence": "Low",
                    "retrieved_chunks": 0
                }

            # Build context
            context = "\n".join([doc.page_content for doc in results])

            # Generate answer with LLM
            if self.llm_client is None:
                # Fallback to simple retrieval if LLM is not available
                fallback_answer = "Based on my knowledge: " + ". ".join([doc.page_content[:150] + "..." for doc in results[:2]])
                return {
                    "answer": fallback_answer,
                    "retrieved_chunks": len(results),
                    "confidence": "Medium"
                }

            try:
                prompt = f"""
                You are Career Compass, a helpful career guidance assistant. Use the context below to answer the question. 
                If the context doesn't contain relevant information, politely say so and suggest related topics you can help with.

                Context from career knowledge base:
                {context}

                Question: {cleaned_question}

                Provide a helpful, concise answer based on the context:
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
                    "confidence": "High" if len(results) >= 2 else "Medium"
                }

            except Exception as e:
                logger.error(f"❌ LLM error: {e}")
                # Fallback to retrieved content
                fallback_answer = "Based on my knowledge: " + ". ".join([doc.page_content[:150] + "..." for doc in results[:2]])
                return {
                    "answer": fallback_answer,
                    "retrieved_chunks": len(results),
                    "confidence": "Medium"
                }

        except Exception as e:
            logger.error(f"❌ Unexpected error in ask_question: {e}")
            return {
                "answer": "I'm experiencing technical difficulties. Please try again later.", 
                "confidence": "Error",
                "retrieved_chunks": 0
            }

    # --- 6️⃣ Health Check ---
    def health_check(self):
        """Check if the RAG system is healthy"""
        try:
            if not self.client or not self.client.is_ready():
                return {"status": "unhealthy", "reason": "Weaviate connection failed"}
            
            if not self.vectorstore:
                return {"status": "degraded", "reason": "Vector store not initialized"}
                
            # Test a simple query
            try:
                test_results = self.vectorstore.similarity_search(query="career", k=1)
                documents_count = len(test_results) if test_results else 0
            except:
                documents_count = 0
                
            return {
                "status": "healthy" if self._is_initialized else "degraded", 
                "initialized": self._is_initialized,
                "vector_store_ready": True,
                "documents_count": documents_count,
                "llm_available": self.llm_client is not None
            }
            
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}

    # --- 7️⃣ Cleanup ---
    def close_connection(self):
        """Close connection to Weaviate"""
        try:
            if self.client:
                self.client.close()
                logger.info("🔌 Weaviate connection closed.")
            self._is_initialized = False
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    def __del__(self):
        """Destructor to ensure connections are closed"""
        self.close_connection()


# Optional: Simple test function
if __name__ == "__main__":
    import sys
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    def test_rag_system():
        """Test the RAG system with a sample question"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")
        
        # Fallback path for different project structure
        if not os.path.exists(csv_path):
            csv_path = os.path.join(base_dir, "app", "data", "final_merged_career_guidance.csv")
        
        print(f"📁 Looking for dataset at: {csv_path}")
        
        system = CareerCompassWeaviate()
        
        if os.path.exists(csv_path):
            print("🚀 Initializing with data...")
            success = system.initialize_system(csv_path)
        else:
            print("⚠️ Initializing minimal system (no data file found)...")
            success = system._initialize_minimal()
        
        if success:
            # Test health check
            health = system.health_check()
            print(f"🏥 System health: {health}")
            
            # Test question
            test_questions = [
                "What skills are important for data scientists?",
                "How do I become a software engineer?",
                "What education is needed for healthcare careers?"
            ]
            
            for question in test_questions:
                print(f"\n❓ Question: {question}")
                start_time = time.time()
                response = system.ask_question(question)
                end_time = time.time()
                
                print(f"✅ Answer: {response['answer']}")
                print(f"📊 Confidence: {response['confidence']}")
                print(f"⏱️ Response time: {end_time - start_time:.2f}s")
                print(f"📚 Chunks retrieved: {response.get('retrieved_chunks', 0)}")
        
        system.close_connection()

    # Run test if executed directly
    test_rag_system()
