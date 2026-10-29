"""Comando único: python3 run.py  (ejecuta todo el caso y se detiene ante cualquier fallo)."""
import sys

from src.pipeline import ejecutar

if __name__ == "__main__":
    sys.exit(ejecutar())
