"""Security utilities for model integrity verification."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex string of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_integrity(
    file_path: Path,
    expected_hash: str | None = None,
    checksums_file: Path | None = None
) -> bool:
    """Verify file integrity using SHA256 checksum.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA256 hash (optional)
        checksums_file: Path to JSON file with checksums (optional)

    Returns:
        True if file is valid, False otherwise

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    actual_hash = calculate_file_hash(file_path)

    # Try to get expected hash from checksums file
    if expected_hash is None and checksums_file and checksums_file.exists():
        try:
            with open(checksums_file) as f:
                checksums = json.load(f)
                expected_hash = checksums.get(str(file_path.name))
        except Exception as e:
            logger.warning(f"Could not load checksums file: {e}")

    # If no expected hash, FAIL CLOSED. Loading an unverified model file is a
    # remote-code-execution vector via pickle/joblib, so refuse rather than
    # silently trusting whatever is on disk.
    if expected_hash is None:
        logger.error(
            f"No checksum available for {file_path.name}. "
            f"Generate checksums with: python -m src.security"
        )
        return False

    # Verify hash
    if actual_hash != expected_hash:
        logger.error(
            f"Checksum mismatch for {file_path.name}!\n"
            f"Expected: {expected_hash}\n"
            f"Actual: {actual_hash}\n"
            f"File may be corrupted or tampered with."
        )
        return False

    logger.info(f"Checksum verified for {file_path.name}")
    return True


def generate_checksums(model_dir: Path, output_file: Path):
    """Generate checksums file for all models in directory.

    Args:
        model_dir: Directory containing model files
        output_file: Output JSON file for checksums
    """
    checksums = {}

    for ext in ['*.pkl', '*.joblib', '*.h5', '*.keras']:
        for file_path in model_dir.glob(ext):
            file_hash = calculate_file_hash(file_path)
            checksums[file_path.name] = file_hash
            logger.info("%s: %s", file_path.name, file_hash)

    with open(output_file, 'w') as f:
        json.dump(checksums, f, indent=2)

    print(f"\nChecksums saved to {output_file}")


if __name__ == "__main__":
    # Generate checksums for models directory
    from pathlib import Path

    from .logging_config import setup_logging

    setup_logging()
    model_dir = Path(__file__).parent.parent / "models"
    output_file = model_dir / "checksums.json"
    generate_checksums(model_dir, output_file)
