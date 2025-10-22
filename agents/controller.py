import logging
from typing import Dict
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class Controller:
    def __init__(self):
        logger.info("Initializing Controller with all agents...")
        self.intent = IntentAgent()
        self.retrieval = RetrievalAgent()
        self.vision = VisionAgent()
        self.reasoning = ReasoningAgent()
        self.memory = MemoryManager()
        logger.info("✓ Controller initialized successfully")
    
    def handle_query(self, query: str) -> Dict:
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            conversation_history = self.memory.get_recent_history()
            
            intent_data = self.intent.predict(query, conversation_history)
            
            retrieved = self.retrieval.search(query, intent_data)
            
            image_descriptions = []
            if retrieved.get("images"):
                logger.info(f"Analyzing {len(retrieved['images'])} images...")
                for img_data in retrieved["images"][:3]:
                    if intent_data.get("intent") == "VISUAL_QUERY":
                        desc = self.vision.analyze_visual_data(img_data["path"], query)
                        image_descriptions.append(desc)
                    else:
                        desc = self.vision.describe(img_data["path"], query)
                        image_descriptions.append(desc)
            
            response = self.reasoning.answer(
                query=query,
                retrieved_data=retrieved,
                image_descriptions=image_descriptions,
                intent_data=intent_data,
                conversation_history=conversation_history
            )
            
            combined_metrics = {
                **intent_data,
                **retrieved.get("metrics", {}),
                **response.get("metrics", {}),
                "quality_score": response.get("quality_score", 0.0)
            }
            
            self.memory.add_turn(
                query=query,
                answer=response["answer"],
                intent_data=intent_data,
                retrieved_data=retrieved,
                metrics=combined_metrics
            )
            
            result = {
                "query": query,
                "answer": response["answer"],
                "intent": intent_data,
                "retrieved_texts": retrieved.get("texts", []),
                "retrieved_images": retrieved.get("images", []),
                "image_descriptions": image_descriptions,
                "metrics": combined_metrics,
                "conversation_summary": self.memory.get_conversation_summary()
            }
            
            logger.info(f"✓ Query processed successfully | Quality: {combined_metrics.get('quality_score', 0):.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in handle_query: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"I encountered a critical error: {str(e)}. Please try again or contact support.",
                "intent": {"intent": "ERROR", "confidence": 0.0},
                "retrieved_texts": [],
                "retrieved_images": [],
                "image_descriptions": [],
                "metrics": {"error": str(e)},
                "conversation_summary": {}
            }
    
    def reset_conversation(self):
        self.memory.clear_memory()
        logger.info("Conversation reset")