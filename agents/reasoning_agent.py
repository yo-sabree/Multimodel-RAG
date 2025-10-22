import logging
import google.generativeai as genai
from typing import Dict, List
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CONTEXT_LENGTH

logger = logging.getLogger(__name__)

class ReasoningAgent:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        logger.info("ReasoningAgent initialized with enhanced config")
    
    def build_context(self, retrieved_data: Dict, image_descriptions: List[Dict] = None) -> str:
        context_parts = []
        total_length = 0
        
        if retrieved_data.get("texts"):
            context_parts.append("=" * 80)
            context_parts.append("RETRIEVED DOCUMENTS")
            context_parts.append("=" * 80)
            
            for i, doc in enumerate(retrieved_data["texts"], 1):
                doc_section = f"""
Document {i}:
├─ Source: {doc.get('source', 'Unknown')}
├─ Page: {doc.get('page', 'N/A')}
├─ Relevance Score: {doc.get('rerank_score', 0):.3f}
└─ Content:
{doc.get('text', '')}
"""
                if total_length + len(doc_section) < MAX_CONTEXT_LENGTH:
                    context_parts.append(doc_section)
                    total_length += len(doc_section)
                else:
                    break
        
        if image_descriptions:
            context_parts.append("\n" + "=" * 80)
            context_parts.append("IMAGE ANALYSIS")
            context_parts.append("=" * 80)
            
            for i, img_desc in enumerate(image_descriptions, 1):
                img_section = f"""
Image {i}:
├─ File: {img_desc.get('image_path', 'Unknown')}
└─ Analysis:
{img_desc.get('description', '')}
"""
                if total_length + len(img_section) < MAX_CONTEXT_LENGTH:
                    context_parts.append(img_section)
                    total_length += len(img_section)
                else:
                    break
        
        return "\n".join(context_parts)
    
    def answer(self, query: str, retrieved_data: Dict, image_descriptions: List[Dict] = None, 
               intent_data: Dict = None, conversation_history: List[Dict] = None) -> Dict:
        try:
            context = self.build_context(retrieved_data, image_descriptions)
            
            history_context = ""
            if conversation_history:
                recent = conversation_history[-2:]
                history_parts = []
                for h in recent:
                    history_parts.append(f"Previous Q: {h.get('query', '')}")
                    history_parts.append(f"Previous A: {h.get('answer', '')[:200]}...")
                history_context = "\n".join(history_parts)
            
            intent_guidance = ""
            if intent_data:
                intent_type = intent_data.get("intent", "")
                if intent_type == "COMPARATIVE_ANALYSIS":
                    intent_guidance = "\n\nFOCUS: Provide a detailed comparison. Create comparison tables if helpful. Highlight key differences and similarities."
                elif intent_type == "SUMMARIZATION":
                    intent_guidance = "\n\nFOCUS: Provide a comprehensive summary. Include key points, main findings, and important details. Use bullet points for clarity."
                elif intent_type == "VISUAL_QUERY":
                    intent_guidance = "\n\nFOCUS: Describe visual elements in detail. Extract all data from charts/graphs. Explain trends and patterns."
                elif intent_type == "COMPLEX_REASONING":
                    intent_guidance = "\n\nFOCUS: Provide step-by-step reasoning. Show your thought process. Connect multiple pieces of information."
                elif intent_type == "PROCEDURAL":
                    intent_guidance = "\n\nFOCUS: Provide clear step-by-step instructions. Number the steps. Include prerequisites and tips."
            
            prompt = f"""You are an expert AI assistant specializing in comprehensive, accurate, and detailed responses.

INSTRUCTIONS:
1. **Accuracy First**: Base your answer ONLY on the provided context
2. **Be Thorough**: Provide detailed explanations with specific evidence
3. **Cite Sources**: Use [Document X, Page Y] or [Image Z] format for all claims
4. **Synthesize Information**: Connect related information from multiple sources
5. **Structure Well**: Use headings, bullet points, and formatting for clarity
6. **Be Honest**: If information is insufficient, clearly state what's missing
7. **Add Value**: Provide insights, implications, and context beyond raw facts

{intent_guidance}

CONVERSATION HISTORY:
{history_context if history_context else "No previous context"}

AVAILABLE CONTEXT:
{context}

USER QUESTION:
{query}

YOUR DETAILED RESPONSE:"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            if not answer or len(answer) < 50:
                answer = "I apologize, but I couldn't generate a sufficient response. Please try rephrasing your question or ensure relevant documents are available."
            
            word_count = len(answer.split())
            has_citations = "[Document" in answer or "[Image" in answer or "[Page" in answer
            
            metrics = {
                "answer_length": len(answer),
                "word_count": word_count,
                "has_citations": has_citations,
                "context_used": len(context),
                "num_sources": len(retrieved_data.get("texts", [])) + len(image_descriptions or [])
            }
            
            logger.info(f"✓ Generated answer: {word_count} words, {metrics['num_sources']} sources")
            
            return {
                "answer": answer,
                "metrics": metrics,
                "quality_score": self._calculate_quality_score(answer, metrics)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}. Please try again.",
                "metrics": {},
                "quality_score": 0.0
            }
    
    def _calculate_quality_score(self, answer: str, metrics: Dict) -> float:
        score = 0.0
        
        if metrics["word_count"] >= 100:
            score += 0.3
        elif metrics["word_count"] >= 50:
            score += 0.2
        
        if metrics["has_citations"]:
            score += 0.3
        
        if metrics["num_sources"] >= 3:
            score += 0.2
        elif metrics["num_sources"] >= 1:
            score += 0.1
        
        structure_markers = ["\n\n", "**", "##", "- ", "1.", "2."]
        structure_count = sum(1 for marker in structure_markers if marker in answer)
        if structure_count >= 3:
            score += 0.2
        elif structure_count >= 1:
            score += 0.1
        
        return min(1.0, score)