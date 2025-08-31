import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.document_ingestor import DocumentIngestor

class TestDocumentIngestor(unittest.TestCase):

    def setUp(self):
        self.ingestor = DocumentIngestor()

    def test_initialization(self):
        self.assertIsNotNone(self.ingestor)
        self.assertIsNotNone(self.ingestor.logger)

    def test_ingest_document(self):
        # This is a complex function that would require a file to be present
        # For a unit test, we'll just test the basic flow with a dummy file
        with open("dummy_doc.txt", "w") as f:
            f.write("This is a test document.")
        
        doc = self.ingestor.ingest_document("dummy_doc.txt")
        self.assertIsNotNone(doc)
        self.assertIn("This is a test document", doc.page_content)
        os.remove("dummy_doc.txt")

if __name__ == '__main__':
    unittest.main()