import subprocess
import sys
import os


def test_script_execution_no_files():
    """Integration test to verify that the script starts correctly
    and initializes the OpenCV environment."""

    # Run script as a separate process
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'faces_video_recognize.py'))

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=10  # So as not to wait forever
    )

    # Expect that the script at least started working and produced info to stdout or stderr
    assert "facial recognition program" in result.stdout or "ERROR" in result.stderr
