"""
Unit tests for PDF embedding pipeline module
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.pdf_embedding_pipeline import (
    PDFEmbeddingPipelineError,
    sanitize_filename,
    _split_markdown_sections,
    _split_paragraph_sentences,
    _build_contextualized_text,
    convert_pdf_to_markdown,
    ensure_collection,
    chunk_document,
    _ensure_dependencies,
    insert_embeddings,
    _build_embedding_function,
    _estimate_average_tokens,
    process_pdf_file,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_DB_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BATCH_SIZE,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function"""

    def test_sanitize_simple_filename(self):
        """Test sanitizing a simple filename"""
        result = sanitize_filename("document.pdf")
        assert result == "document"

    def test_sanitize_filename_with_spaces(self):
        """Test sanitizing filename with spaces"""
        result = sanitize_filename("my document.pdf")
        assert result == "my_document"

    def test_sanitize_filename_with_special_chars(self):
        """Test sanitizing filename with special characters"""
        result = sanitize_filename("doc@#$%ument!.pdf")
        assert result == "doc_ument"

    def test_sanitize_filename_with_path(self):
        """Test sanitizing filename with path"""
        result = sanitize_filename("/path/to/document.pdf")
        assert result == "document"

    def test_sanitize_empty_filename(self):
        """Test sanitizing empty or invalid filename"""
        result = sanitize_filename("@#$%.pdf")
        assert result == "document"

    def test_sanitize_filename_preserves_hyphens_underscores(self):
        """Test that hyphens and underscores are preserved"""
        result = sanitize_filename("my-doc_name.pdf")
        assert result == "my-doc_name"


class TestSplitMarkdownSections:
    """Tests for _split_markdown_sections function"""

    def test_split_single_section(self):
        """Test splitting markdown with single section"""
        markdown = "# Header\nContent here"
        sections = _split_markdown_sections(markdown)
        assert len(sections) == 1
        assert sections[0][0] == "Header"
        assert sections[0][1] == "Content here"

    def test_split_multiple_sections(self):
        """Test splitting markdown with multiple sections"""
        markdown = "# Section 1\nContent 1\n\n## Section 2\nContent 2"
        sections = _split_markdown_sections(markdown)
        assert len(sections) == 2
        assert sections[0][0] == "Section 1"
        assert sections[1][0] == "Section 2"

    def test_split_with_introduction(self):
        """Test splitting markdown with content before first header"""
        markdown = "Intro text\n# Header\nContent"
        sections = _split_markdown_sections(markdown)
        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert sections[0][1] == "Intro text"

    def test_split_empty_sections_filtered(self):
        """Test that empty sections are filtered out"""
        markdown = "# Header 1\n\n# Header 2\nContent"
        sections = _split_markdown_sections(markdown)
        assert len(sections) == 1
        assert sections[0][0] == "Header 2"

    def test_split_various_header_levels(self):
        """Test splitting with different header levels"""
        markdown = "# H1\nC1\n## H2\nC2\n### H3\nC3"
        sections = _split_markdown_sections(markdown)
        assert len(sections) == 3
        assert sections[0][0] == "H1"
        assert sections[1][0] == "H2"
        assert sections[2][0] == "H3"


class TestSplitParagraphSentences:
    """Tests for _split_paragraph_sentences function"""

    def test_split_simple_sentences(self):
        """Test splitting simple sentences"""
        paragraph = "First sentence. Second sentence. Third sentence."
        sentences = _split_paragraph_sentences(paragraph)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."

    def test_split_with_multiple_spaces(self):
        """Test splitting with multiple spaces"""
        paragraph = "First sentence.  Second   sentence."
        sentences = _split_paragraph_sentences(paragraph)
        assert len(sentences) == 2

    def test_split_empty_paragraph(self):
        """Test splitting empty paragraph"""
        sentences = _split_paragraph_sentences("")
        assert sentences == []

    def test_split_with_exclamation_question(self):
        """Test splitting with exclamation and question marks"""
        paragraph = "Hello! How are you? I'm fine."
        sentences = _split_paragraph_sentences(paragraph)
        assert len(sentences) == 3

    def test_split_preserves_sentence_content(self):
        """Test that sentence content is preserved"""
        paragraph = "This is a test. With some content."
        sentences = _split_paragraph_sentences(paragraph)
        assert "This is a test." in sentences
        assert "With some content." in sentences


class TestBuildContextualizedText:
    """Tests for _build_contextualized_text function"""

    def test_build_contextualized_text(self):
        """Test building contextualized text"""
        result = _build_contextualized_text("doc1", "Section A", "Content here")
        assert "Document: doc1" in result
        assert "Section: Section A" in result
        assert "Content here" in result

    def test_contextualized_text_format(self):
        """Test that contextualized text has correct format"""
        result = _build_contextualized_text("test", "intro", "text")
        lines = result.split("\n")
        assert lines[0] == "Document: test"
        assert lines[1] == "Section: intro"
        assert lines[2] == ""
        assert lines[3] == "text"


class TestConvertPdfToMarkdown:
    """Tests for convert_pdf_to_markdown function"""

    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_convert_pdf_success(self, mock_ensure_deps, mock_converter_class):
        """Test successful PDF conversion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake PDF file
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf")
            
            output_dir = Path(tmpdir) / "output"
            
            # Mock converter
            mock_converter = MagicMock()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = "# Test\nContent"
            mock_converter.convert.return_value = mock_result
            mock_converter_class.return_value = mock_converter
            
            result = convert_pdf_to_markdown(tmpdir, str(output_dir))
            
            assert len(result) == 1
            assert result[0].endswith("test.md")
            assert output_dir.exists()

    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_convert_pdf_invalid_source_dir(self, mock_ensure_deps):
        """Test conversion with invalid source directory"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Source directory does not exist"):
            convert_pdf_to_markdown("/nonexistent/path", "/output")

    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_convert_pdf_no_pdfs_found(self, mock_ensure_deps, mock_converter_class):
        """Test conversion when no PDFs are found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PDFEmbeddingPipelineError, match="No PDF files found"):
                convert_pdf_to_markdown(tmpdir, tmpdir)

    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_convert_pdf_handles_conversion_error(self, mock_ensure_deps, mock_converter_class):
        """Test that conversion errors are handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf")
            
            mock_converter = MagicMock()
            mock_converter.convert.side_effect = Exception("Conversion failed")
            mock_converter_class.return_value = mock_converter
            
            result = convert_pdf_to_markdown(tmpdir, tmpdir)
            
            # Should return empty list when all conversions fail
            assert result == []


class TestEnsureCollection:
    """Tests for ensure_collection function"""

    def test_ensure_collection_existing(self):
        """Test getting existing collection"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_embedding_fn = MagicMock()
        
        collection, status = ensure_collection(mock_client, "test_collection", mock_embedding_fn)
        
        assert collection == mock_collection
        assert status == "existing"
        mock_client.get_collection.assert_called_once()

    def test_ensure_collection_create_new(self):
        """Test creating new collection"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embedding_fn = MagicMock()
        
        collection, status = ensure_collection(mock_client, "new_collection", mock_embedding_fn)
        
        assert collection == mock_collection
        assert status == "created"
        mock_client.get_or_create_collection.assert_called_once()


class TestChunkDocument:
    """Tests for chunk_document function"""

    def test_chunk_document_basic(self):
        """Test basic document chunking"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Section 1\n\nParagraph 1.\n\nParagraph 2.")
            f.flush()
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            
            try:
                chunks = chunk_document(f.name, mock_tokenizer, max_tokens=100)
                
                assert len(chunks) > 0
                assert all('text' in chunk for chunk in chunks)
                assert all('contextualized_text' in chunk for chunk in chunks)
                assert all('section_header' in chunk for chunk in chunks)
                assert all('token_count' in chunk for chunk in chunks)
            finally:
                Path(f.name).unlink()

    def test_chunk_document_nonexistent_file(self):
        """Test chunking nonexistent document"""
        mock_tokenizer = MagicMock()
        
        with pytest.raises(PDFEmbeddingPipelineError, match="Document path does not exist"):
            chunk_document("/nonexistent/file.md", mock_tokenizer)

    def test_chunk_document_respects_max_tokens(self):
        """Test that chunking respects max token limit"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Section\n\n" + "Word. " * 100)
            f.flush()
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1] * 10  # 10 tokens per call
            
            try:
                chunks = chunk_document(f.name, mock_tokenizer, max_tokens=50)
                
                # All chunks should respect token limit
                assert all(chunk['token_count'] <= 50 for chunk in chunks)
            finally:
                Path(f.name).unlink()

    def test_chunk_document_handles_long_paragraphs(self):
        """Test chunking with paragraphs exceeding max tokens"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            # Create a very long paragraph
            long_paragraph = "Sentence. " * 200
            f.write(f"# Section\n\n{long_paragraph}")
            f.flush()
            
            mock_tokenizer = MagicMock()
            # Simulate paragraph being too long
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1] * (len(text) // 2)
            
            try:
                chunks = chunk_document(f.name, mock_tokenizer, max_tokens=100)
                
                # Should split long paragraph into multiple chunks
                assert len(chunks) > 1
            finally:
                Path(f.name).unlink()


class TestEnsureDependencies:
    """Tests for _ensure_dependencies function"""

    @patch('src.pdf_embedding_pipeline.DocumentConverter', None)
    def test_ensure_dependencies_missing_docling(self):
        """Test error when docling is missing"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Missing required dependencies"):
            _ensure_dependencies()

    @patch('src.pdf_embedding_pipeline.chromadb', None)
    def test_ensure_dependencies_missing_chromadb(self):
        """Test error when chromadb is missing"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Missing required dependencies"):
            _ensure_dependencies()

    @patch('src.pdf_embedding_pipeline.SentenceTransformer', None)
    def test_ensure_dependencies_missing_sentence_transformers(self):
        """Test error when sentence-transformers is missing"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Missing required dependencies"):
            _ensure_dependencies()

    @patch('src.pdf_embedding_pipeline.AutoTokenizer', None)
    def test_ensure_dependencies_missing_transformers(self):
        """Test error when transformers is missing"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Missing required dependencies"):
            _ensure_dependencies()


class TestInsertEmbeddings:
    """Tests for insert_embeddings function"""

    def test_insert_embeddings_single_batch(self):
        """Test inserting embeddings in single batch"""
        mock_collection = MagicMock()
        chunks = [
            {
                'text': 'chunk1',
                'contextualized_text': 'ctx1',
                'token_count': 10,
                'section_header': 'Section 1'
            },
            {
                'text': 'chunk2',
                'contextualized_text': 'ctx2',
                'token_count': 15,
                'section_header': 'Section 2'
            }
        ]
        
        result = insert_embeddings(chunks, mock_collection, "doc1", batch_size=10)
        
        assert result['inserted_count'] == 2
        assert result['batch_size'] == 10
        mock_collection.add.assert_called_once()

    def test_insert_embeddings_multiple_batches(self):
        """Test inserting embeddings in multiple batches"""
        mock_collection = MagicMock()
        chunks = [
            {
                'text': f'chunk{i}',
                'contextualized_text': f'ctx{i}',
                'token_count': 10,
                'section_header': 'Section'
            }
            for i in range(15)
        ]
        
        result = insert_embeddings(chunks, mock_collection, "doc1", batch_size=5)
        
        assert result['inserted_count'] == 15
        # Should be called 3 times (5 + 5 + 5)
        assert mock_collection.add.call_count == 3

    def test_insert_embeddings_metadata_format(self):
        """Test that metadata is formatted correctly"""
        mock_collection = MagicMock()
        chunks = [{
            'text': 'original text',
            'contextualized_text': 'contextualized',
            'token_count': 20,
            'section_header': 'Header'
        }]
        
        insert_embeddings(chunks, mock_collection, "test_doc", batch_size=10)
        
        call_args = mock_collection.add.call_args
        metadatas = call_args[1]['metadatas']
        
        assert metadatas[0]['filename'] == 'test_doc'
        assert metadatas[0]['chunk_index'] == 0
        assert metadatas[0]['token_count'] == 20
        assert metadatas[0]['section_header'] == 'Header'
        assert metadatas[0]['original_text'] == 'original text'


class TestBuildEmbeddingFunction:
    """Tests for _build_embedding_function function"""

    @patch('src.pdf_embedding_pipeline.embedding_functions')
    def test_build_embedding_function_success(self, mock_embedding_fns):
        """Test building embedding function"""
        mock_fn = MagicMock()
        mock_embedding_fns.SentenceTransformerEmbeddingFunction.return_value = mock_fn
        
        result = _build_embedding_function("test-model")
        
        assert result == mock_fn
        mock_embedding_fns.SentenceTransformerEmbeddingFunction.assert_called_once_with(
            model_name="test-model"
        )

    @patch('src.pdf_embedding_pipeline.embedding_functions', None)
    def test_build_embedding_function_unavailable(self):
        """Test error when embedding functions unavailable"""
        with pytest.raises(PDFEmbeddingPipelineError, match="embedding functions are unavailable"):
            _build_embedding_function("test-model")


class TestEstimateAverageTokens:
    """Tests for _estimate_average_tokens function"""

    def test_estimate_average_tokens(self):
        """Test calculating average tokens"""
        chunks = [
            {'token_count': 100},
            {'token_count': 200},
            {'token_count': 300}
        ]
        
        avg = _estimate_average_tokens(chunks)
        
        assert avg == 200.0

    def test_estimate_average_tokens_empty(self):
        """Test with empty chunks list"""
        avg = _estimate_average_tokens([])
        assert avg == 0.0

    def test_estimate_average_tokens_single_chunk(self):
        """Test with single chunk"""
        chunks = [{'token_count': 150}]
        avg = _estimate_average_tokens(chunks)
        assert avg == 150.0


class TestProcessPdfFile:
    """Tests for process_pdf_file function"""

    @patch('src.pdf_embedding_pipeline.embedding_functions')
    @patch('src.pdf_embedding_pipeline.chromadb')
    @patch('src.pdf_embedding_pipeline.SentenceTransformer')
    @patch('src.pdf_embedding_pipeline.AutoTokenizer')
    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_process_pdf_file_success(
        self,
        mock_ensure_deps,
        mock_converter_class,
        mock_tokenizer_class,
        mock_transformer_class,
        mock_chromadb,
        mock_embedding_fns
    ):
        """Test successful PDF processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake PDF
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf")
            
            # Mock converter
            mock_converter = MagicMock()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = "# Test\n\nContent here."
            mock_converter.convert.return_value = mock_result
            mock_converter_class.return_value = mock_converter
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # Mock transformer
            mock_transformer_class.return_value = MagicMock()
            
            # Mock embedding functions
            mock_embedding_fn = MagicMock()
            mock_embedding_fns.SentenceTransformerEmbeddingFunction.return_value = mock_embedding_fn
            
            # Mock ChromaDB
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            result = process_pdf_file(
                pdf_path=str(pdf_path),
                output_dir=tmpdir
            )
            
            assert result['document_name'] == 'test.pdf'
            assert 'markdown_path' in result
            assert 'chunk_count' in result
            assert 'inserted_count' in result
            assert result['collection_name'] == DEFAULT_COLLECTION_NAME

    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_process_pdf_file_invalid_path(self, mock_ensure_deps):
        """Test processing with invalid PDF path"""
        with pytest.raises(PDFEmbeddingPipelineError, match="Invalid PDF file path"):
            process_pdf_file("/nonexistent/file.pdf")

    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_process_pdf_file_not_pdf(self, mock_ensure_deps):
        """Test processing with non-PDF file"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(PDFEmbeddingPipelineError, match="Invalid PDF file path"):
                process_pdf_file(f.name)

    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_process_pdf_file_conversion_failure(self, mock_ensure_deps, mock_converter_class):
        """Test handling of conversion failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf")
            
            mock_converter = MagicMock()
            mock_converter.convert.side_effect = Exception("Conversion error")
            mock_converter_class.return_value = mock_converter
            
            with pytest.raises(PDFEmbeddingPipelineError, match="Failed to convert PDF to markdown"):
                process_pdf_file(str(pdf_path))

    @patch('src.pdf_embedding_pipeline.chromadb')
    @patch('src.pdf_embedding_pipeline.SentenceTransformer')
    @patch('src.pdf_embedding_pipeline.AutoTokenizer')
    @patch('src.pdf_embedding_pipeline.DocumentConverter')
    @patch('src.pdf_embedding_pipeline._ensure_dependencies')
    def test_process_pdf_file_no_chunks_generated(
        self,
        mock_ensure_deps,
        mock_converter_class,
        mock_tokenizer_class,
        mock_transformer_class,
        mock_chromadb
    ):
        """Test error when no chunks are generated"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_text("fake pdf")
            
            # Mock converter to return empty markdown
            mock_converter = MagicMock()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = ""
            mock_converter.convert.return_value = mock_result
            mock_converter_class.return_value = mock_converter
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = []
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_transformer_class.return_value = MagicMock()
            
            with pytest.raises(PDFEmbeddingPipelineError, match="No chunks generated"):
                process_pdf_file(str(pdf_path), output_dir=tmpdir)


class TestConstants:
    """Tests for module constants"""

    def test_default_constants_defined(self):
        """Test that all default constants are defined"""
        assert DEFAULT_EMBEDDING_MODEL is not None
        assert DEFAULT_DB_PATH is not None
        assert DEFAULT_COLLECTION_NAME is not None
        assert DEFAULT_MAX_TOKENS > 0
        assert DEFAULT_BATCH_SIZE > 0

    def test_default_embedding_model_format(self):
        """Test that default embedding model has expected format"""
        assert "/" in DEFAULT_EMBEDDING_MODEL
        assert "sentence-transformers" in DEFAULT_EMBEDDING_MODEL


class TestPDFEmbeddingPipelineError:
    """Tests for custom exception"""

    def test_exception_can_be_raised(self):
        """Test that custom exception can be raised"""
        with pytest.raises(PDFEmbeddingPipelineError):
            raise PDFEmbeddingPipelineError("Test error")

    def test_exception_message(self):
        """Test that exception message is preserved"""
        try:
            raise PDFEmbeddingPipelineError("Custom message")
        except PDFEmbeddingPipelineError as e:
            assert str(e) == "Custom message"

# Made with Bob
