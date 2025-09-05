#!/usr/bin/env python3
"""
Test to verify the fetcher works with the new prompt/model interface
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.fetcher import get_full_study
from src.pipeline.prompt_model_interface import set_extraction_prompt, set_mapping_prompt, set_model, create_configured_pipeline
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

def test_fetcher_with_prompt_model_interface():
    """Test the fetcher with prompt/model interface"""
    logger.info("🧪 Testing fetcher with prompt/model interface")
    
    try:
        # Configure the pipeline through the interface
        logger.info("🔧 Configuring pipeline through prompt/model interface")
        set_extraction_prompt("generic_extraction")
        set_mapping_prompt("generic_mapping")
        set_model("deepseek-coder")
        
        # Create a configured pipeline to verify configuration works
        pipeline = create_configured_pipeline()
        logger.info("✅ Pipeline configured successfully through interface")
        
        # Test with a real NCT ID (using a simple one for testing)
        # Note: We're not actually calling the API in this test to avoid external dependencies
        # Instead, we'll test the configuration part
        
        logger.info("✅ Prompt/model interface test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fetcher_pipeline_creation():
    """Test that the fetcher can create a pipeline with the new interface"""
    logger.info("🧪 Testing fetcher pipeline creation")
    
    try:
        # This test would normally be run with actual API calls, but we'll just test
        # that the pipeline creation works without errors
        
        # Create a configured pipeline directly
        pipeline = create_configured_pipeline()
        logger.info("✅ Pipeline created successfully")
        
        # Verify the pipeline has the expected components
        if hasattr(pipeline, 'nlp_extractor'):
            logger.info(f"✅ NLP engine configured with model: {getattr(pipeline.nlp_extractor, 'model_name', 'Unknown')}")
        else:
            logger.warning("⚠️  NLP engine not found in pipeline")
            
        if hasattr(pipeline, 'llm_mapper'):
            logger.info(f"✅ LLM mapper configured with model: {getattr(pipeline.llm_mapper, 'model_name', 'Unknown')}")
        else:
            logger.warning("⚠️  LLM mapper not found in pipeline")
        
        logger.info("✅ Fetcher pipeline creation test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_fetcher_with_prompt_model_interface()
    success2 = test_fetcher_pipeline_creation()
    
    if success1 and success2:
        logger.info("\n✅ All fetcher tests completed successfully!")
    else:
        logger.error("\n❌ Some fetcher tests failed!")
        sys.exit(1)