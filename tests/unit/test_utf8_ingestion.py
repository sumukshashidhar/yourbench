"""Test that UTF-8 text files are handled correctly in ingestion."""
import tempfile
from pathlib import Path
import pytest
from markitdown import MarkItDown
from omegaconf import OmegaConf
from yourbench.pipeline.ingestion import _convert_file


def test_utf8_text_file_ingestion():
    """Test that .txt files with UTF-8 characters are read correctly.
    
    Regression test for issue #186 where UTF-8 encoded .txt files caused
    UnicodeDecodeError due to MarkItDown incorrectly detecting ASCII encoding.
    """
    # Create a mock config
    config = OmegaConf.create({
        'pipeline': {
            'ingestion': {
                'supported_file_extensions': ['.txt', '.md'],
                'llm_ingestion': False
            }
        }
    })
    
    processor = MarkItDown()
    
    # Create test file with UTF-8 characters that produce 0xe2 byte
    # (the byte mentioned in issue #186)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
        test_content = 'Test UTF-8: • bullet – en dash — em dash € euro © copyright'
        f.write(test_content)
        temp_path = Path(f.name)
    
    try:
        # This should NOT raise UnicodeDecodeError
        result = _convert_file(temp_path, config, processor)
        
        assert result is not None, 'Result should not be None'
        assert '•' in result, 'Bullet point (U+2022) should be preserved'
        assert '–' in result, 'En dash (U+2013) should be preserved' 
        assert '—' in result, 'Em dash (U+2014) should be preserved'
        assert '€' in result, 'Euro sign (U+20AC) should be preserved'
        assert '©' in result, 'Copyright (U+00A9) should be preserved'
        
    finally:
        temp_path.unlink()


def test_utf8_text_extension_variants():
    """Test that both .txt and .text extensions work with UTF-8."""
    config = OmegaConf.create({
        'pipeline': {
            'ingestion': {
                'supported_file_extensions': ['.txt', '.text'],
                'llm_ingestion': False
            }
        }
    })
    
    processor = MarkItDown()
    test_content = 'UTF-8 test: • – — € ©'
    
    for suffix in ['.txt', '.text']:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=suffix, delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)
        
        try:
            result = _convert_file(temp_path, config, processor)
            assert result is not None
            assert '•' in result, f'{suffix} should handle UTF-8 correctly'
        finally:
            temp_path.unlink()
