"""
Parsers module - input file parsers for various formats.

This module provides parsers that read problem instances from files
and construct Problem objects.

Available Parsers:
-----------------
- Parser: Abstract base class for custom parsers
- KasirzadehParser: Parser for Kasirzadeh crew scheduling instances

Usage:
------
>>> from opencg.parsers import KasirzadehParser
>>>
>>> parser = KasirzadehParser()
>>> problem = parser.parse("path/to/instance")
>>> print(problem.summary())

Custom Parsers:
--------------
To create a custom parser, subclass Parser and implement parse():

>>> from opencg.parsers import Parser
>>>
>>> class MyParser(Parser):
...     def parse(self, path: str) -> Problem:
...         # Read files, construct network, etc.
...         return problem
"""

from opencg.parsers.base import Parser
from opencg.parsers.kasirzadeh import KasirzadehParser

__all__ = [
    "Parser",
    "KasirzadehParser",
]
