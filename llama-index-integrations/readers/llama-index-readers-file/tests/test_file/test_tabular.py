import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from tenacity import RetryError

from llama_index.core.schema import Document
from llama_index.core.readers.file.base import get_default_fs
from llama_index.readers.file import (
    DocxReader,
    PDFReader,
)  # assuming the code is in pdf_reader.py


class TestPDFReader:
    @pytest.fixture()
    def pdf_reader(self):
        return PDFReader()

    @pytest.fixture()
    def pdf_reader_full_doc(self):
        return PDFReader(return_full_document=True)

    @pytest.fixture()
    def mock_pdf(self):
        # Create a mock PDF with two pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Content of page 1"

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Content of page 2"

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.page_labels = ["1", "2"]

        return mock_pdf

    def test_init(self):
        """Test PDFReader initialization."""
        reader = PDFReader()
        assert reader.return_full_document is False
        assert reader.is_remote is False

        reader_full = PDFReader(return_full_document=True)
        assert reader_full.return_full_document is True

    @patch("pypdf.PdfReader")
    def test_load_data_single_page(self, mock_pdf_reader, pdf_reader):
        """Test loading a single page PDF."""
        # Setup mock
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.page_labels = ["1"]
        mock_pdf_reader.return_value = mock_pdf

        # Test
        file_path = Path("test.pdf")
        fs = get_default_fs()

        with patch("builtins.open", mock_open(read_data=b"pdf content")):
            docs = pdf_reader.load_data(file_path, fs=fs)

        assert len(docs) == 1
        assert docs[0].text == "Test content"
        assert docs[0].metadata == {"page_label": "1", "file_name": "test.pdf"}

    @patch("pypdf.PdfReader")
    def test_load_data_multiple_pages(self, mock_pdf_reader, pdf_reader, mock_pdf):
        """Test loading a multi-page PDF."""
        mock_pdf_reader.return_value = mock_pdf
        file_path = Path("test.pdf")
        fs = get_default_fs()

        with patch("builtins.open", mock_open(read_data=b"pdf content")):
            docs = pdf_reader.load_data(file_path, fs=fs)

        assert len(docs) == 2
        assert docs[0].text == "Content of page 1"
        assert docs[1].text == "Content of page 2"
        assert docs[0].metadata == {"page_label": "1", "file_name": "test.pdf"}
        assert docs[1].metadata == {"page_label": "2", "file_name": "test.pdf"}

    @patch("pypdf.PdfReader")
    def test_load_data_full_document(
        self, mock_pdf_reader, pdf_reader_full_doc, mock_pdf
    ):
        """Test loading a PDF as a single document."""
        mock_pdf_reader.return_value = mock_pdf
        file_path = Path("test.pdf")
        fs = get_default_fs()

        with patch("builtins.open", mock_open(read_data=b"pdf content")):
            docs = pdf_reader_full_doc.load_data(file_path, fs=fs)

        assert len(docs) == 1
        assert docs[0].text == "Content of page 1\nContent of page 2"
        assert docs[0].metadata == {"file_name": "test.pdf"}

    @patch("pypdf.PdfReader")
    def test_load_data_with_extra_info(self, mock_pdf_reader, pdf_reader, mock_pdf):
        """Test loading a PDF with extra metadata."""
        mock_pdf_reader.return_value = mock_pdf
        file_path = Path("test.pdf")
        extra_info = {"author": "Test Author", "year": 2024}
        fs = get_default_fs()

        with patch("builtins.open", mock_open(read_data=b"pdf content")):
            docs = pdf_reader.load_data(file_path, extra_info=extra_info, fs=fs)

        assert len(docs) == 2
        expected_metadata = {
            "page_label": "1",
            "file_name": "test.pdf",
            "author": "Test Author",
            "year": 2024,
        }
        assert docs[0].metadata == expected_metadata

    def test_missing_pypdf(self):
        """Test handling of missing pypdf dependency."""
        pdf_reader = PDFReader()
        file_path = Path("test.pdf")

        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises(RetryError) as exc_info:
                pdf_reader.load_data(file_path)

            # Check that the wrapped exception is an ImportError with our expected message
            assert isinstance(exc_info.value.last_attempt.exception(), ImportError)
            assert "pypdf is required to read PDF files" in str(
                exc_info.value.last_attempt.exception()
            )

    @patch("pypdf.PdfReader")
    def test_non_path_input(self, mock_pdf_reader, pdf_reader, mock_pdf):
        """Test handling of string input instead of Path."""
        mock_pdf_reader.return_value = mock_pdf
        file_path = "test.pdf"
        fs = get_default_fs()

        with patch("builtins.open", mock_open(read_data=b"pdf content")):
            docs = pdf_reader.load_data(file_path, fs=fs)

        assert len(docs) == 2
        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)


class TestDocxReader:
    @pytest.fixture()
    def docx_reader(self):
        """Basic DocxReader fixture."""
        return DocxReader()

    @pytest.fixture()
    def sample_text(self):
        """Sample document text."""
        return "This is a sample document text with multiple paragraphs.\n\nSecond paragraph here."

    def test_init(self):
        """Test DocxReader initialization."""
        reader = DocxReader()
        assert reader.is_remote is False

        # Test with explicit is_remote
        reader_remote = DocxReader(is_remote=True)
        assert reader_remote.is_remote is True

    def test_load_data_local_file(self, docx_reader, sample_text):
        """Test loading a local docx file."""
        file_path = Path("test.docx")

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path)

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == sample_text
        assert docs[0].metadata == {"file_name": "test.docx"}

    def test_load_data_with_extra_info(self, docx_reader, sample_text):
        """Test loading a file with extra metadata."""
        file_path = Path("test.docx")
        extra_info = {"author": "Test Author", "date": "2024-01-01"}

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path, extra_info=extra_info)

        assert len(docs) == 1
        assert docs[0].text == sample_text
        assert docs[0].metadata == {
            "file_name": "test.docx",
            "author": "Test Author",
            "date": "2024-01-01",
        }

    def test_load_data_with_remote_fs(self, docx_reader, sample_text):
        """Test loading a file using a remote filesystem."""
        file_path = Path("test.docx")
        mock_fs = Mock()
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        mock_fs.open.return_value = mock_file

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path, fs=mock_fs)

        assert len(docs) == 1
        assert docs[0].text == sample_text
        mock_fs.open.assert_called_once_with(str(file_path))
        mock_process.assert_called_once_with(mock_file)

    def test_load_data_with_string_path(self, docx_reader, sample_text):
        """Test loading a file using a string path instead of Path object."""
        file_path = "test.docx"

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path)

        assert len(docs) == 1
        assert docs[0].metadata["file_name"] == "test.docx"
        mock_process.assert_called_once()

    def test_missing_docx2txt(self, docx_reader):
        """Test handling of missing docx2txt dependency."""
        file_path = Path("test.docx")

        with patch.dict("sys.modules", {"docx2txt": None}):
            with pytest.raises(ImportError) as exc_info:
                docx_reader.load_data(file_path)

            assert "docx2txt is required to read Microsoft Word files" in str(
                exc_info.value
            )

    def test_load_data_empty_metadata(self, docx_reader, sample_text):
        """Test loading a file with empty metadata."""
        file_path = Path("test.docx")
        extra_info = {}

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path, extra_info=extra_info)

        assert len(docs) == 1
        assert docs[0].metadata == {"file_name": "test.docx"}

    def test_load_data_none_metadata(self, docx_reader, sample_text):
        """Test loading a file with None metadata."""
        file_path = Path("test.docx")
        extra_info = None

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = sample_text
            docs = docx_reader.load_data(file_path, extra_info=extra_info)

        assert len(docs) == 1
        assert docs[0].metadata == {"file_name": "test.docx"}

    @pytest.mark.parametrize(
        "text_content",
        [
            "",
            "Simple text",
            "Multi\nline\ntext",
            "Text with специальные characters",
            "Text with\ttabs and spaces    ",
        ],
    )
    def test_load_data_various_content(self, docx_reader, text_content):
        """Test loading files with various types of content."""
        file_path = Path("test.docx")

        with patch("docx2txt.process") as mock_process:
            mock_process.return_value = text_content
            docs = docx_reader.load_data(file_path)

        assert len(docs) == 1
        assert docs[0].text == text_content

    def test_process_error_handling(self, docx_reader):
        """Test handling of docx2txt.process errors."""
        file_path = Path("test.docx")

        with patch("docx2txt.process") as mock_process:
            mock_process.side_effect = Exception("Failed to process document")

            with pytest.raises(Exception) as exc_info:
                docx_reader.load_data(file_path)

            assert "Failed to process document" in str(exc_info.value)
