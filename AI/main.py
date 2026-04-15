import sys
import os
import time
import subprocess

def run_orchestrator():
    """
    Launch the AI Intruder Detection pipeline concurrently.
    Starts watchdog.py, waits 2 seconds for webcam/model initialization,
    then starts face_processor.py.
    """
    # Calculate the absolute path of the directory containing main.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Safely construct the absolute paths to the scripts
    watchdog_path = os.path.join(BASE_DIR, "src", "watchdog.py")
    processor_path = os.path.join(BASE_DIR, "src", "face_processor.py")

    print("============================================================")
    print("  STARTING AI INTRUDER DETECTION SYSTEM MAIN ORCHESTRATOR")
    print("============================================================")
    
    print(f"[main.py] Launching {watchdog_path}...")
    # Start the watchdog using the absolute path
    watchdog_process = subprocess.Popen([sys.executable, watchdog_path])
    
    # Warmup delay for webcam and ML models
    print("[main.py] Waiting 2 seconds for systems to initialize...")
    time.sleep(2)
    
    print(f"[main.py] Launching {processor_path}...")
    # Start the face processor using the absolute path
    processor_process = subprocess.Popen([sys.executable, processor_path])
    
    try:
        # Wait for processes to complete naturally
        watchdog_process.wait()
        processor_process.wait()
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\n============================================================")
        print("  KEYBOARD INTERRUPT DETECTED: Shutting down entire pipeline")
        print("============================================================")
        
        print("[main.py] Terminating watchdog (Releasing Webcam)...")
        watchdog_process.terminate()
        
        print("[main.py] Terminating face_processor...")
        processor_process.terminate()
        
        # Ensure they are fully stopped
        watchdog_process.wait()
        processor_process.wait()
        
        print("[main.py] All processes cleanly terminated. Exiting.")

if __name__ == "__main__":
    run_orchestrator()