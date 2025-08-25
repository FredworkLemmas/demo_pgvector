import attrs
import magic
import os
import pypandoc

from pathlib import Path
from typing import Iterable

from lib.constants import INTERNAL_WORKDIR
from lib.interfaces import SourceConversionTool


@attrs.define
class SourceIdentifier:
    path: str

    def _magic_identifier(self) -> str:
        return magic.from_file(self.path, mime=True)

    def identifier(self) -> str:
        return self._magic_identifier()


class BaseSourceConversionTool(object):
    converted_suffix: str = None

    def convertable_types(self) -> list[str]:
        return []

    def convert(self, source: str) -> str:
        raise NotImplementedError('convert method not implemented')

    @classmethod
    def converted_path(self, source: str) -> str:
        """Return an available filename in the workdir for the converted file"""
        filename = os.path.basename(source)
        filename_without_suffix = (
            filename
            if '.' not in filename
            else '.'.join(filename.split('.')[0:-1])
        )
        filename_with_new_suffix = (
            f'{filename_without_suffix}.{self.converted_suffix}'
        )
        if not os.path.exists(
            os.path.join(INTERNAL_WORKDIR, filename_with_new_suffix)
        ):
            return os.path.join(INTERNAL_WORKDIR, filename_with_new_suffix)
        counter = 0
        while True:
            counter += 1
            filename = (
                f'{filename_without_suffix}_{counter}.{self.converted_suffix}'
            )
            if not os.path.exists(os.path.join(INTERNAL_WORKDIR, filename)):
                return os.path.join(INTERNAL_WORKDIR, filename)


class EPUBSourceConversionTool(BaseSourceConversionTool):
    converted_suffix = 'md'

    @staticmethod
    def convertible_types() -> list[str]:
        return ['application/epub+zip']

    def convert(self, source: str) -> str:
        output_path = self.converted_path(source)
        return self.convert_epub_advanced(source, output_path)

    def convert_epub_to_markdown_pypandoc(
        self,
        epub_path: str,
        output_md_path: str,
        extract_media: bool = True,
        extra_args: list = None,
    ) -> str:
        """
        Convert an EPUB file to Markdown using pypandoc.

        Args:
            epub_path: Path to the input EPUB file
            output_md_path: Path to the output Markdown file
            extract_media: If True, extract images and media files
            extra_args: Additional pandoc arguments (optional)

        Returns:
            Path to the generated Markdown file

        Raises:
            FileNotFoundError: If EPUB file doesn't exist
            RuntimeError: If pypandoc/pandoc is not properly installed
            OSError: If conversion fails
        """

        # Check if pypandoc is available
        try:
            pypandoc.get_pandoc_version()
        except OSError as e:
            raise RuntimeError(
                f'Pandoc not found. Install with: pip install pypandoc-binary\n'
                f'Or install pandoc system-wide. Error: {e}'
            )

        epub_path = Path(epub_path)
        output_md_path = Path(output_md_path)

        # Check if input file exists
        if not epub_path.exists():
            raise FileNotFoundError(f'EPUB file not found: {epub_path}')

        # Create output directory if it doesn't exist
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        # Build extra arguments list
        pandoc_args = [
            '--wrap=none',  # Don't wrap lines
            # '--atx-headers',  # Use ATX-style headers (# ## ###)
            '--standalone',  # Create standalone document
        ]

        # Handle media extraction
        if extract_media:
            media_dir = output_md_path.parent / (output_md_path.stem + '_media')
            pandoc_args.append(f'--extract-media={media_dir}')

        # Add custom arguments if provided
        if extra_args:
            pandoc_args.extend(extra_args)

        try:
            # Convert EPUB to Markdown
            _ = pypandoc.convert_file(
                str(epub_path),
                'markdown',
                format='epub',
                outputfile=str(output_md_path),
                extra_args=pandoc_args,
            )

            return str(output_md_path)

        except Exception as e:
            raise OSError(f'Conversion failed: {e}')

    def convert_epub_advanced(self, epub_path: str, output_md_path: str) -> str:
        """
        Convert EPUB to Markdown with advanced formatting options.
        """

        advanced_args = [
            '--markdown-headings=atx',  # Use # style headers
            '--wrap=preserve',  # Preserve original line wrapping
            '--tab-stop=2',  # Set tab stops to 2 spaces
            '--eol=lf',  # Use LF line endings
            '--strip-comments',  # Remove HTML comments
            '--reference-links',  # Use reference-style links
            '--columns=80',  # Set line width for wrapping
        ]

        return self.convert_epub_to_markdown_pypandoc(
            epub_path,
            output_md_path,
            extract_media=True,
            extra_args=advanced_args,
        )


class SourceConverter:
    raw_sources: Iterable[str] = None
    processed_sources: Iterable[str] = None
    type_to_conversion_class: dict[str, SourceConversionTool] | None = None
    conversion_tools: list[SourceConversionTool] = [EPUBSourceConversionTool]

    def __init__(self, sources: Iterable[str] = None):
        self.raw_sources = sources or []

        # init type to conversion class mapping
        self.type_to_conversion_class = {}
        for conversion_tool in self.conversion_tools:
            for identifier in conversion_tool.convertible_types():
                self.type_to_conversion_class[identifier] = conversion_tool

    def needs_conversion(self, source: str) -> bool:
        identifier = SourceIdentifier(source).identifier()
        # print(f'file type identifier: {identifier}')
        return identifier not in ['text/plain']

    def is_convertible(self, source: str) -> bool:
        identifier = SourceIdentifier(source).identifier()
        # print(f'file type identifier: {identifier}')
        return identifier in self.type_to_conversion_class

    def convert(self, source: str) -> str:
        converter_class = self.type_to_conversion_class[
            SourceIdentifier(source).identifier()
        ]
        return converter_class().convert(source)

    def ingestion_ready_sources(self) -> list[str]:
        if self.processed_sources:
            return self.processed_sources

        self.processed_sources = []
        converter = SourceConverter()
        # print(f'raw sources: {self.raw_sources}')
        for file in self.raw_sources:
            if not converter.needs_conversion(file):
                # print(f'adding unconverted file: {file}')
                self.processed_sources.append(file)
                continue
            if converter.is_convertible(file):
                # print(f'adding converted file: {file}')
                converted_file = converter.convert(file)
                self.processed_sources.append(converted_file)
        return self.processed_sources
