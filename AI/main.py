import sys
import os
import time
import subprocess

def run_orchestrator():
    """
    Launch the AI Intruder Detection CORE SERVER.
    This runs the face processor and database worker concurrently in the background.
    It does NOT connect to cameras. It waits for Edge cameras to feed it data.
    """
    # Calculate the absolute path of the directory containing main.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Safely construct the absolute paths to the scripts
    processor_path = os.path.join(BASE_DIR, "src", "face_processor.py")
    db_worker_path = os.path.join(BASE_DIR, "src", "database", "db_worker.py")

    print("============================================================")
    print("  STARTING AI CORE SERVER")
    print("  (Waiting for camera feeds...)")
    print("============================================================")
    
    print(f"[main.py] Launching {processor_path}...")
    # Start the face processor
    processor_process = subprocess.Popen([sys.executable, processor_path])
    
    print(f"[main.py] Launching {db_worker_path}...")
    # Start the database worker
    db_worker_process = subprocess.Popen([sys.executable, db_worker_path])
    
    try:
        # Wait for processes to complete naturally
        processor_process.wait()
        db_worker_process.wait()
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\n============================================================")
        print("  KEYBOARD INTERRUPT DETECTED: Shutting down Core Server")
        print("============================================================")
        
        print("[main.py] Terminating face_processor...")
        processor_process.terminate()

        print("[main.py] Terminating db_worker...")
        db_worker_process.terminate()
        
        # Ensure they are fully stopped
        processor_process.wait()
        db_worker_process.wait()
        
        print("[main.py] All processes cleanly terminated. Exiting.")

if __name__ == "__main__":
    run_orchestrator()