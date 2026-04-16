import sys
import os

# Get the absolute path to the AI folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Append the root AI/ directory
sys.path.append(BASE_DIR)
# Append the src/ directory so watchdog can find config.py!
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.watchdog import run_watchdog

if __name__ == "__main__":
    print("--- Starting Edge Camera (Property 2 | Live) ---")
    run_watchdog(property_id=2, video_source=0)
