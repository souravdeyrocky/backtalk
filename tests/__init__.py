"""Test package init -- runs when `python -m unittest discover -s
tests` imports this package. Historically this was the ONLY thing
that redirected real-state modules away from this machine's real
files; it no longer is. See tests/isolation.py's module docstring for
why that assumption failed on 2026-09-21, and why every test file that
touches real-state modules now calls isolation.ensure_isolated()
explicitly at its own top level instead of relying solely on this file
having been imported.

This call is kept only as a convenience for the common case (running
the full suite, or any pattern that happens to import a file that
also calls it) -- it is redundant with, never a substitute for, each
test file's own explicit call.
"""
from tests import isolation

isolation.ensure_isolated()
