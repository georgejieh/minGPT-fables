"""Script to clean Aesop's Fables text file in place, line-by-line logic.

This script processes 'aesop_fables_clean.txt' located in 'data/cleaned/'.
It standardizes formatting line-by-line, removes commentary, and ensures
each fable starts with '###' followed by title and body, with correct spacing.
"""

import os

def is_title_line(line: str) -> bool:
    """Check if a line is a fable title (all uppercase letters, spaces or punctuation).

    Args:
        line: Line string to check.

    Returns:
        True if line looks like a title, False otherwise.
    """
    stripped = line.strip()
    return (
        stripped.isupper()
        and any(c.isalpha() for c in stripped)
    )

def clean_fables(file_path: str) -> None:
    """Clean Aesop's Fables text file line by line and overwrite in place.

    Args:
        file_path: Path to the input partially cleaned fables file.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    cleaned_lines = []
    previous_line_type = None  # can be "title", "text", "empty", or None

    for line in lines:
        stripped = line.rstrip("\n")

        # Check if indented line (commentary)
        if stripped.startswith(" ") or stripped.startswith("\t"):
            continue

        # Check if title line
        if is_title_line(stripped):
            if not cleaned_lines or cleaned_lines[-1] != "###":
                cleaned_lines.append("###")
            cleaned_lines.append(stripped)
            previous_line_type = "title"
            continue

        # Check if empty line
        if not stripped.strip():
            if previous_line_type in {"title", "text"} and (not cleaned_lines or cleaned_lines[-1] != ""):
                cleaned_lines.append("")
            previous_line_type = "empty"
            continue

        # Regular text line (body)
        cleaned_lines.append(stripped)
        previous_line_type = "text"

    # Join lines and ensure final newline at end
    final_text = "\n".join(cleaned_lines).strip() + "\n"

    with open(file_path, "w", encoding="utf-8") as out_file:
        out_file.write(final_text)

    print(f"Finished cleaning. File overwritten at {file_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    FILE_PATH = os.path.join(ROOT_DIR, "data", "cleaned", "aesop_fables_clean.txt")

    clean_fables(FILE_PATH)
