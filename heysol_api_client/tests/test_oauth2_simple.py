"""
Simple OAuth2 test for HeySol API client.

This script tests the OAuth2 implementation with the correct Google OAuth2
configuration for HeySol.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.oauth2 import OAuth2ClientCredentialsAuthenticator, OAuth2Error
from heysol.exceptions import HeySolError, AuthenticationError

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oauth2_simple_test.log")
    ]
)
logger = logging.getLogger(__name__)


def test_oauth2_setup():
    """Test OAuth2 setup with HeySol's Google OAuth2 configuration."""
    logger.info("Testing OAuth2 setup with HeySol Google OAuth2...")

    try:
        # Get OAuth2 credentials from environment
        client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
        client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

        logger.info(f"Client ID: {client_id}")
        logger.info(f"Client Secret: {'[SET]' if client_secret else '[NOT SET]'}")

        if not client_id or not client_secret:
            logger.error("OAuth2 credentials not found in environment variables")
            logger.info("Please set COREAI_OAUTH2_CLIENT_ID and COREAI_OAUTH2_CLIENT_SECRET")
            return False

        # Create OAuth2 authenticator
        config = HeySolConfig(
            oauth2_client_id=client_id,
            oauth2_client_secret=client_secret,
            log_level="DEBUG",
            log_to_file=True,
            log_file_path="oauth2_simple_debug.log"
        )

        oauth2_auth = OAuth2ClientCredentialsAuthenticator(config)

        # Try to authenticate
        logger.info("Attempting OAuth2 client credentials authentication...")
        tokens = oauth2_auth.authenticate()

        logger.info("✅ OAuth2 authentication successful!")
        logger.info(f"Access token: {tokens.access_token[:20]}...")
        logger.info(f"Token type: {tokens.token_type}")
        logger.info(f"Expires in: {tokens.expires_in} seconds")

        return True

    except Exception as e:
        logger.error(f"❌ OAuth2 authentication failed: {e}")
        return False


def test_client_with_oauth2():
    """Test HeySol client with OAuth2 authentication."""
    logger.info("Testing HeySol client with OAuth2...")

    try:
        # Get OAuth2 credentials from environment
        client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
        client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.error("OAuth2 credentials not found")
            return False

        # Create client with OAuth2
        config = HeySolConfig(
            oauth2_client_id=client_id,
            oauth2_client_secret=client_secret,
            log_level="DEBUG",
            log_to_file=True,
            log_file_path="oauth2_client_debug.log"
        )

        client = HeySolClient(
            config=config,
            use_oauth2=True
        )

        logger.info("✅ HeySol client with OAuth2 initialized successfully!")

        # Test basic functionality
        logger.info("Testing basic client functionality...")

        # Test getting spaces
        spaces = client.get_spaces()
        logger.info(f"✅ Successfully retrieved {len(spaces)} spaces")

        # Test creating a space
        space_name = f"test_oauth2_space_{int(__import__('time').time())}"
        space_id = client.create_space(space_name, "Test space for OAuth2")
        logger.info(f"✅ Successfully created space: {space_id}")

        # Test deleting the space
        delete_result = client.delete_space(space_id)
        logger.info(f"✅ Successfully deleted space: {delete_result}")

        return True

    except Exception as e:
        logger.error(f"❌ Client with OAuth2 failed: {e}")
        return False


def main():
    """Main function to run OAuth2 tests."""
    logger.info("Starting HeySol OAuth2 testing...")

    print("\n" + "="*60)
    print("HEYSOL OAUTH2 AUTHENTICATION TEST")
    print("="*60)

    # Test OAuth2 setup
    print("\n1. Testing OAuth2 Setup...")
    oauth2_success = test_oauth2_setup()

    if oauth2_success:
        print("✅ OAuth2 setup successful!")

        # Test client with OAuth2
        print("\n2. Testing Client with OAuth2...")
        client_success = test_client_with_oauth2()

        if client_success:
            print("✅ Client with OAuth2 successful!")
            print("\n🎉 ALL OAUTH2 TESTS PASSED!")
            print("The HeySol API client is ready to use with OAuth2 authentication!")
        else:
            print("❌ Client with OAuth2 failed")
    else:
        print("❌ OAuth2 setup failed")
        print("\nTo fix this:")
        print("1. Get the OAuth2 client secret from HeySol")
        print("2. Set COREAI_OAUTH2_CLIENT_SECRET environment variable")
        print("3. Run the test again")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()