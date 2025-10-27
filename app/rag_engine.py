def ask_question(self, question):
    """Retrieve + Generate an answer using RAG with better error handling"""
    print(f"🔍 DEBUG: Entering ask_question with: {question}")
    
    try:
        # Ensure system is initialized
        print(f"🔍 DEBUG: Checking initialization - _is_initialized: {self._is_initialized}")
        if not self._is_initialized:
            print("🔍 DEBUG: System not initialized")
            return {
                "answer": "System is still initializing. Please try again in a moment.", 
                "confidence": "Low",
                "retrieved_chunks": 0
            }

        print(f"🔍 DEBUG: Checking vectorstore - exists: {self.vectorstore is not None}")
        if not self.vectorstore:
            print("🔍 DEBUG: Vectorstore is None")
            return {
                "answer": "Knowledge base is not available at the moment.", 
                "confidence": "Error",
                "retrieved_chunks": 0
            }

        # Validate question
        if not question or not question.strip():
            print("🔍 DEBUG: Empty question")
            return {
                "answer": "Please provide a valid question.", 
                "confidence": "Low",
                "retrieved_chunks": 0
            }

        cleaned_question = question.strip()
        print(f"🔍 DEBUG: Processing question: {cleaned_question}")

        # Retrieve context with error handling
        try:
            print("🔍 DEBUG: Attempting similarity search...")
            results = self.vectorstore.similarity_search(
                query=cleaned_question, 
                k=3
            )
            print(f"🔍 DEBUG: Retrieved {len(results)} results")
        except Exception as e:
            print(f"🔍 DEBUG: Vector search error: {e}")
            return {
                "answer": "Error searching knowledge base. Please try again.", 
                "confidence": "Error",
                "retrieved_chunks": 0
            }

        if not results:
            print("🔍 DEBUG: No results found")
            return {
                "answer": "I don't have enough information about this topic in my knowledge base.", 
                "confidence": "Low",
                "retrieved_chunks": 0
            }

        # Build context
        context = "\n".join([doc.page_content for doc in results])
        print(f"🔍 DEBUG: Context built, length: {len(context)}")

        # Check if LLM client is available
        print(f"🔍 DEBUG: Checking LLM client - exists: {self.llm_client is not None}")
        if self.llm_client is None:
            print("🔍 DEBUG: LLM client not available, using fallback")
            # Fallback to simple retrieval if LLM is not available
            fallback_answer = "Based on my knowledge: " + ". ".join([doc.page_content[:150] + "..." for doc in results[:2]])
            return {
                "answer": fallback_answer,
                "retrieved_chunks": len(results),
                "confidence": "Medium"
            }

        # Generate answer with LLM
        try:
            print("🔍 DEBUG: Generating answer with LLM...")
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
            print(f"🔍 DEBUG: Successfully generated answer: {final_answer[:100]}...")
            
            return {
                "answer": final_answer,
                "retrieved_chunks": len(results),
                "confidence": "High" if len(results) >= 2 else "Medium"
            }

        except Exception as e:
            print(f"🔍 DEBUG: LLM generation error: {e}")
            # Fallback to retrieved content
            fallback_answer = "Based on my knowledge: " + ". ".join([doc.page_content[:150] + "..." for doc in results[:2]])
            return {
                "answer": fallback_answer,
                "retrieved_chunks": len(results),
                "confidence": "Medium"
            }

    except Exception as e:
        print(f"🔍 DEBUG: UNEXPECTED ERROR in ask_question: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error stack
        return {
            "answer": "I'm experiencing technical difficulties. Please try again later.", 
            "confidence": "Error",
            "retrieved_chunks": 0
        }
