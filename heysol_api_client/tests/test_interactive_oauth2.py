"""
Interactive OAuth2 test for HeySol API client.

This script demonstrates the complete interactive OAuth2 flow:
1. Opens a browser for user authentication
2. Handles the callback automatically
3. Tests API functionality with OAuth2 tokens
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
from heysol.oauth2_interactive import InteractiveOAuth2Authenticator
from heysol.exceptions import HeySolError, AuthenticationError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("interactive_oauth2_test.log")
    ]
)
logger = logging.getLogger(__name__)


class InteractiveOAuth2Tester:
    """Test interactive OAuth2 authentication and API functionality."""

    def __init__(self):
        """Initialize the OAuth2 tester."""
        self.client: HeySolClient | None = None
        self.oauth2_auth: InteractiveOAuth2Authenticator | None = None

    def setup_interactive_oauth2(self) -> bool:
        """Setup interactive OAuth2 authentication."""
        try:
            logger.info("Setting up interactive OAuth2 authentication...")

            # Get OAuth2 credentials from environment
            client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
            client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

            if not client_id:
                logger.error("OAuth2 client ID not found in environment variables")
                logger.info("Please set COREAI_OAUTH2_CLIENT_ID")
                return False

            logger.info(f"Using OAuth2 client ID: {client_id[:20]}...")

            # Create OAuth2 authenticator
            config = HeySolConfig(
                oauth2_client_id=client_id,
                oauth2_client_secret=client_secret,  # May be None for public clients
                log_level="INFO",
                log_to_file=True,
                log_file_path="interactive_oauth2_debug.log"
            )

            self.oauth2_auth = InteractiveOAuth2Authenticator(config)
            logger.info("Interactive OAuth2 authenticator created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup interactive OAuth2: {e}")
            return False

    def perform_interactive_auth(self) -> bool:
        """Perform interactive OAuth2 authentication."""
        try:
            if not self.oauth2_auth:
                logger.error("OAuth2 authenticator not initialized")
                return False

            logger.info("Starting interactive OAuth2 authentication...")
            logger.info("A browser window will open for authentication.")

            # Perform interactive authorization
            tokens = self.oauth2_auth.authorize_interactive(
                scope="openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
            )

            logger.info("✅ Interactive OAuth2 authentication successful!")
            logger.info(f"Access token: {tokens.access_token[:20]}...")
            logger.info(f"Token type: {tokens.token_type}")
            logger.info(f"Expires in: {tokens.expires_in} seconds")

            return True

        except Exception as e:
            logger.error(f"❌ Interactive OAuth2 authentication failed: {e}")
            return False

    def setup_client_with_oauth2(self) -> bool:
        """Setup HeySol client with OAuth2 authentication."""
        try:
            if not self.oauth2_auth:
                logger.error("OAuth2 authenticator not initialized")
                return False

            logger.info("Setting up HeySol client with OAuth2...")

            # Get OAuth2 credentials from environment
            client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
            client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

            # Create client with OAuth2
            config = HeySolConfig(
                oauth2_client_id=client_id,
                oauth2_client_secret=client_secret,
                log_level="INFO",
                log_to_file=True,
                log_file_path="interactive_client_debug.log"
            )

            self.client = HeySolClient(
                config=config,
                use_oauth2=True
            )

            logger.info("✅ HeySol client with OAuth2 initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup client with OAuth2: {e}")
            return False

    def test_api_functionality(self) -> bool:
        """Test API functionality with OAuth2 tokens."""
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False

            logger.info("Testing API functionality with OAuth2...")

            # Test getting spaces
            logger.info("📋 Testing get_spaces...")
            spaces = self.client.get_spaces()
            logger.info(f"✅ Successfully retrieved {len(spaces)} spaces")

            # Test creating a space
            logger.info("🏗️ Testing create_space...")
            space_name = f"test_interactive_oauth2_{int(__import__('time').time())}"
            space_id = self.client.create_space(space_name, "Test space for interactive OAuth2")
            logger.info(f"✅ Successfully created space: {space_id}")

            # Test memory ingestion
            logger.info("💾 Testing memory_ingestion...")
            message = "This is a test message from interactive OAuth2 authentication"
            result = self.client.ingest_memory(message, space_id=space_id)
            logger.info("✅ Successfully ingested memory")

            # Test memory search
            logger.info("🔍 Testing memory_search...")
            search_results = self.client.search_memory("test message", space_id=space_id)
            logger.info(f"✅ Successfully searched memory: {len(search_results)} results")

            # Test deleting the space
            logger.info("🗑️ Testing delete_space...")
            delete_result = self.client.delete_space(space_id)
            logger.info(f"✅ Successfully deleted space: {delete_result}")

            return True

        except Exception as e:
            logger.error(f"❌ API functionality test failed: {e}")
            return False

    def run_complete_test(self) -> bool:
        """Run the complete interactive OAuth2 test."""
        logger.info("🚀 Starting complete interactive OAuth2 test...")

        # Step 1: Setup OAuth2
        logger.info("Step 1: Setting up OAuth2...")
        if not self.setup_interactive_oauth2():
            logger.error("Failed to setup OAuth2")
            return False

        # Step 2: Perform interactive authentication
        logger.info("Step 2: Performing interactive authentication...")
        if not self.perform_interactive_auth():
            logger.error("Failed to perform interactive authentication")
            return False

        # Step 3: Setup client with OAuth2
        logger.info("Step 3: Setting up client with OAuth2...")
        if not self.setup_client_with_oauth2():
            logger.error("Failed to setup client with OAuth2")
            return False

        # Step 4: Test API functionality
        logger.info("Step 4: Testing API functionality...")
        if not self.test_api_functionality():
            logger.error("Failed to test API functionality")
            return False

        logger.info("🎉 All tests passed! Interactive OAuth2 is working perfectly!")
        return True


def main():
    """Main function to run interactive OAuth2 tests."""
    logger.info("Starting HeySol Interactive OAuth2 Testing...")

    print("\n" + "="*70)
    print("HEYSOL INTERACTIVE OAUTH2 AUTHENTICATION TEST")
    print("="*70)
    print("\nThis test will:")
    print("1. Open a browser for Google OAuth2 authentication")
    print("2. Handle the callback automatically")
    print("3. Test API functionality with OAuth2 tokens")
    print("\nMake sure you have:")
    print("- COREAI_OAUTH2_CLIENT_ID set in your environment")
    print("- A valid Google account for authentication")
    print("- Internet connection")
    print("\n" + "="*70)

    # Check prerequisites
    client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
    if not client_id:
        print("\n❌ ERROR: COREAI_OAUTH2_CLIENT_ID not found in environment variables")
        print("Please set COREAI_OAUTH2_CLIENT_ID and try again.")
        return

    print(f"\n✅ Client ID: {client_id[:20]}...")
    print("✅ Ready to start interactive OAuth2 test!")
    print("\nPress Enter to continue...")
    input()

    # Run the test
    tester = InteractiveOAuth2Tester()
    success = tester.run_complete_test()

    # Print results
    print("\n" + "="*70)
    if success:
        print("🎉 SUCCESS: Interactive OAuth2 test completed successfully!")
        print("\nThe HeySol API client now supports:")
        print("✅ Interactive OAuth2 authentication")
        print("✅ Browser-based authentication")
        print("✅ Automatic callback handling")
        print("✅ Full API functionality with OAuth2 tokens")
        print("\nYou can now use OAuth2 authentication in your applications!")
    else:
        print("❌ FAILURE: Interactive OAuth2 test failed")
        print("\nCheck the logs for details:")
        print("- interactive_oauth2_test.log")
        print("- interactive_oauth2_debug.log")
        print("- interactive_client_debug.log")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()