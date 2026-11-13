"""Compatibility entry point for the current standard-base final audit."""
from importlib import import_module


def main():
    return import_module('analysis.32_20_4_bounded_comparison.finalize').main()


if __name__ == '__main__':
    main()
