"""
Parser base module - abstract base class for parsers.

All parsers should inherit from Parser and implement the parse() method.
This ensures a consistent interface across different file formats.

Design Notes:
------------
- Parsers are stateless (no instance data beyond configuration)
- parse() returns a Problem object
- Parsers can optionally validate input
- Parsers can have configuration options
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from opencg.core.problem import Problem


@dataclass
class ParserConfig:
    """
    Configuration options for parsers.

    Attributes:
        validate: Whether to validate the parsed problem
        verbose: Whether to print progress messages
        encoding: File encoding (default UTF-8)
        options: Additional parser-specific options
    """
    validate: bool = True
    verbose: bool = False
    encoding: str = "utf-8"
    options: Dict[str, Any] = field(default_factory=dict)


class Parser(ABC):
    """
    Abstract base class for problem parsers.

    A Parser reads files and constructs a Problem object.

    Subclasses must implement:
    - parse(): Read files and return a Problem

    Optional overrides:
    - can_parse(): Check if parser can handle a path
    - get_format_name(): Return human-readable format name

    Example:
        >>> class MyFormatParser(Parser):
        ...     def parse(self, path: str) -> Problem:
        ...         # Read files from path
        ...         # Construct network, resources, etc.
        ...         # Return Problem object
        ...         pass
        ...
        ...     def get_format_name(self) -> str:
        ...         return "MyFormat"
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize parser with configuration.

        Args:
            config: Parser configuration (uses defaults if None)
        """
        self.config = config or ParserConfig()

    @abstractmethod
    def parse(self, path: Union[str, Path]) -> Problem:
        """
        Parse files and return a Problem.

        This is the main method subclasses must implement.

        Args:
            path: Path to file or directory containing problem data

        Returns:
            Constructed Problem object

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If files are malformed
        """
        pass

    def can_parse(self, path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given path.

        Default implementation checks if path exists.
        Subclasses can override for format-specific checks.

        Args:
            path: Path to check

        Returns:
            True if this parser can handle the path
        """
        path = Path(path)
        return path.exists()

    def get_format_name(self) -> str:
        """
        Return human-readable format name.

        Returns:
            Format name (e.g., "Kasirzadeh", "TSPLIB")
        """
        return self.__class__.__name__.replace("Parser", "")

    def _log(self, message: str) -> None:
        """
        Print a log message if verbose mode is enabled.

        Args:
            message: Message to print
        """
        if self.config.verbose:
            print(f"[{self.get_format_name()}] {message}")

    def _read_file(self, path: Union[str, Path]) -> str:
        """
        Read a file with configured encoding.

        Args:
            path: Path to file

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        with open(path, 'r', encoding=self.config.encoding) as f:
            return f.read()

    def _read_lines(self, path: Union[str, Path]) -> list:
        """
        Read file lines with configured encoding.

        Args:
            path: Path to file

        Returns:
            List of lines (stripped of whitespace)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        with open(path, 'r', encoding=self.config.encoding) as f:
            return [line.strip() for line in f]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ParserRegistry:
    """
    Registry for parsers.

    Allows automatic parser selection based on file format.

    Example:
        >>> registry = ParserRegistry()
        >>> registry.register(KasirzadehParser)
        >>> registry.register(TSPLIBParser)
        >>>
        >>> # Auto-detect and parse
        >>> problem = registry.parse("path/to/instance")
    """

    def __init__(self):
        """Create an empty registry."""
        self._parsers: Dict[str, type] = {}

    def register(
        self,
        parser_class: type,
        name: Optional[str] = None
    ) -> None:
        """
        Register a parser class.

        Args:
            parser_class: Parser class to register
            name: Optional name (uses class name if not provided)
        """
        if name is None:
            name = parser_class.__name__
        self._parsers[name] = parser_class

    def get(self, name: str) -> Optional[type]:
        """
        Get a parser class by name.

        Args:
            name: Parser name

        Returns:
            Parser class or None
        """
        return self._parsers.get(name)

    def parse(
        self,
        path: Union[str, Path],
        parser_name: Optional[str] = None,
        **config_options
    ) -> Problem:
        """
        Parse a problem, auto-detecting format if needed.

        Args:
            path: Path to problem files
            parser_name: Specific parser to use (auto-detect if None)
            **config_options: Options passed to ParserConfig

        Returns:
            Parsed Problem

        Raises:
            ValueError: If no suitable parser found
        """
        config = ParserConfig(**config_options)
        path = Path(path)

        if parser_name:
            parser_class = self._parsers.get(parser_name)
            if parser_class is None:
                raise ValueError(f"Unknown parser: {parser_name}")
            parser = parser_class(config)
            return parser.parse(path)

        # Try each parser
        for name, parser_class in self._parsers.items():
            parser = parser_class(config)
            if parser.can_parse(path):
                return parser.parse(path)

        raise ValueError(f"No parser found for: {path}")

    def list_parsers(self) -> list:
        """Get list of registered parser names."""
        return list(self._parsers.keys())


# Global registry
_registry = ParserRegistry()


def register_parser(parser_class: type, name: Optional[str] = None) -> None:
    """Register a parser in the global registry."""
    _registry.register(parser_class, name)


def get_parser(name: str) -> Optional[type]:
    """Get a parser from the global registry."""
    return _registry.get(name)


def parse(path: Union[str, Path], **options) -> Problem:
    """Parse using the global registry with auto-detection."""
    return _registry.parse(path, **options)
