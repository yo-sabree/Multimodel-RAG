import os
import logging
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import re
from typing import List, Dict
from config import (
    REPORTS_DIR, IMAGES_DIR, TEXT_INDEX_PATH, IMAGE_INDEX_PATH,
    TEXT_META_PATH, IMAGE_META_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, TOP_K_RERANK, TEXT_EMBED_MODEL, RERANK_MODEL,
    MIN_CHUNK_SIZE, SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class RetrievalAgent:
    def __init__(self):
        logger.info("Initializing RetrievalAgent with advanced models...")
        
        self.text_model = SentenceTransformer(TEXT_EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_index = None
        self.image_index = None
        self.text_meta = []
        self.image_meta = []
        
        logger.info("RetrievalAgent initialized successfully")
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def semantic_chunking(self, text: str, source: str, page: int) -> List[Dict]:
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length > CHUNK_SIZE and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                if len(chunk_text) >= MIN_CHUNK_SIZE:
                    chunks.append({
                        "source": source,
                        "page": page,
                        "text": self.clean_text(chunk_text)
                    })
                
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "source": source,
                    "page": page,
                    "text": self.clean_text(chunk_text)
                })
        
        return chunks
    
    def extract_texts_from_pdfs(self) -> List[Dict]:
        docs = []
        pdf_files = list(REPORTS_DIR.glob("*.pdf"))
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and len(text.strip()) > MIN_CHUNK_SIZE:
                            chunks = self.semantic_chunking(text, pdf_file.name, page_num)
                            docs.extend(chunks)
                
                logger.info(f"✓ {pdf_file.name}: {len([d for d in docs if d['source'] == pdf_file.name])} chunks")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        logger.info(f"Total documents extracted: {len(docs)}")
        return docs
    
    def extract_images(self) -> List[Dict]:
        images = []
        for img_file in IMAGES_DIR.glob("*"):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                images.append({"path": str(img_file), "name": img_file.name})
        logger.info(f"Found {len(images)} images")
        return images
    
    def build_text_index(self):
        docs = self.extract_texts_from_pdfs()
        if not docs:
            logger.warning("No documents found to index")
            return
        
        logger.info("Generating embeddings for text documents...")
        embeddings = self.text_model.encode(
            [doc["text"] for doc in docs],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        embeddings = np.array(embeddings).astype("float32")
        
        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        self.text_index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(docs) // 10))
        self.text_index.train(embeddings)
        self.text_index.add(embeddings)
        self.text_meta = docs
        
        faiss.write_index(self.text_index, str(TEXT_INDEX_PATH))
        np.save(TEXT_META_PATH, np.array(self.text_meta, dtype=object))
        
        logger.info(f"✓ Text index built: {len(docs)} documents indexed")
    
    def build_image_index(self):
        images = self.extract_images()
        if not images:
            logger.warning("No images found to index")
            return
        
        logger.info("Generating embeddings for images...")
        embeddings = []
        
        for img_data in images:
            try:
                image = Image.open(img_data["path"]).convert("RGB")
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    emb = self.clip_model.get_image_features(**inputs).cpu().numpy()[0]
                    emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error processing image {img_data['name']}: {e}")
                images.remove(img_data)
        
        if embeddings:
            embeddings = np.array(embeddings).astype("float32")
            self.image_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.image_index.add(embeddings)
            self.image_meta = images
            
            faiss.write_index(self.image_index, str(IMAGE_INDEX_PATH))
            np.save(IMAGE_META_PATH, np.array(self.image_meta, dtype=object))
            
            logger.info(f"✓ Image index built: {len(images)} images indexed")
    
    def load_indexes(self):
        if TEXT_INDEX_PATH.exists():
            self.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
            self.text_meta = np.load(TEXT_META_PATH, allow_pickle=True).tolist()
            if hasattr(self.text_index, 'nprobe'):
                self.text_index.nprobe = 10
            logger.info(f"✓ Loaded text index: {len(self.text_meta)} documents")
        
        if IMAGE_INDEX_PATH.exists():
            self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            self.image_meta = np.load(IMAGE_META_PATH, allow_pickle=True).tolist()
            logger.info(f"✓ Loaded image index: {len(self.image_meta)} images")
    
    def rerank_results(self, query: str, documents: List[Dict]) -> List[Dict]:
        if not documents:
            return []
        
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Reranked {len(documents)} documents, top score: {reranked[0]['rerank_score']:.3f}")
        return reranked[:TOP_K_RERANK]
    
    def search(self, query: str, intent_data: dict = None) -> Dict:
        results = {"texts": [], "images": [], "metrics": {}}
        
        top_k = TOP_K_RETRIEVAL
        if intent_data:
            if intent_data.get("complexity") == "high":
                top_k = TOP_K_RETRIEVAL * 2
            elif intent_data.get("complexity") == "low":
                top_k = max(5, TOP_K_RETRIEVAL // 2)
        
        if self.text_index and self.text_meta:
            try:
                query_emb = self.text_model.encode(
                    query,
                    normalize_embeddings=True
                ).astype("float32").reshape(1, -1)
                
                k = min(top_k, len(self.text_meta))
                scores, indices = self.text_index.search(query_emb, k)
                
                retrieved_docs = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.text_meta) and idx >= 0:
                        doc = self.text_meta[idx].copy()
                        doc["similarity_score"] = float(score)
                        if score >= SIMILARITY_THRESHOLD:
                            retrieved_docs.append(doc)
                
                if retrieved_docs:
                    reranked_docs = self.rerank_results(query, retrieved_docs)
                    results["texts"] = reranked_docs
                    
                    results["metrics"]["text_retrieved"] = len(retrieved_docs)
                    results["metrics"]["text_after_rerank"] = len(reranked_docs)
                    results["metrics"]["avg_similarity"] = np.mean([d["similarity_score"] for d in retrieved_docs])
                    results["metrics"]["avg_rerank_score"] = np.mean([d["rerank_score"] for d in reranked_docs])
                    
            except Exception as e:
                logger.error(f"Text search error: {e}")
        
        if self.image_index and self.image_meta and (not intent_data or intent_data.get("requires_images")):
            try:
                inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
                with torch.no_grad():
                    query_emb = self.clip_model.get_text_features(**inputs).cpu().numpy()
                    query_emb = query_emb / np.linalg.norm(query_emb)
                    query_emb = query_emb.astype("float32")
                
                k = min(3, len(self.image_meta))
                scores, indices = self.image_index.search(query_emb, k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.image_meta) and idx >= 0:
                        img = self.image_meta[idx].copy()
                        img["similarity_score"] = float(score)
                        if score >= SIMILARITY_THRESHOLD:
                            results["images"].append(img)
                
                results["metrics"]["images_retrieved"] = len(results["images"])
                
            except Exception as e:
                logger.error(f"Image search error: {e}")
        
        logger.info(f"Retrieved: {len(results['texts'])} texts, {len(results['images'])} images")
        return results