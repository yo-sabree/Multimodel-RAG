import logging
import sys
from agents.retrieval_agent import RetrievalAgent
from config import LOG_LEVEL, LOG_FORMAT, REPORTS_DIR, IMAGES_DIR

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     🚀 HIGH-ACCURACY MULTIMODAL RAG SYSTEM                    ║
║     Powered by Gemini 2.0 Flash                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_data_availability():
    pdf_count = len(list(REPORTS_DIR.glob("*.pdf")))
    image_count = len([f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    print(f"\n📊 Data Status:")
    print(f"   ├─ PDFs found: {pdf_count}")
    print(f"   └─ Images found: {image_count}")
    
    if pdf_count == 0 and image_count == 0:
        print("\n⚠️  WARNING: No documents or images found!")
        print(f"   Please add PDFs to: {REPORTS_DIR}")
        print(f"   Please add images to: {IMAGES_DIR}")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    return pdf_count, image_count

def main():
    print_banner()
    logger.info("="*80)
    logger.info("Starting RAG System Index Builder")
    logger.info("="*80)
    
    pdf_count, image_count = check_data_availability()
    
    print("\n🔧 Initializing Retrieval Agent...")
    retrieval = RetrievalAgent()
    
    if pdf_count > 0:
        print(f"\n📄 Building text index from {pdf_count} PDFs...")
        print("   This may take a few minutes depending on document size...")
        retrieval.build_text_index()
        print("   ✓ Text index created successfully!")
    else:
        print("\n⚠️  Skipping text index (no PDFs found)")
    
    if image_count > 0:
        print(f"\n🖼️  Building image index from {image_count} images...")
        retrieval.build_image_index()
        print("   ✓ Image index created successfully!")
    else:
        print("\n⚠️  Skipping image index (no images found)")
    
    print("\n" + "="*70)
    print("✅ INDEX BUILDING COMPLETE!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   Run: streamlit run streamlit_app.py")
    print("   Then open: http://localhost:8501")
    print("\n" + "="*70)
    
    logger.info("Index building completed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)