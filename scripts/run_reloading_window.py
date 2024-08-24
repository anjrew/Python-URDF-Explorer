import os
import sys
import time
import subprocess
import shutil

# Name of your binary (as it appears in PATH)
BINARY_NAME = "urdf-viz"
BINARY_PATH = shutil.which(BINARY_NAME)
WMCTRL_PATH = shutil.which("wmctrl")

if not BINARY_PATH:
    print(f"Error: '{BINARY_NAME}' not found in PATH. Please make sure it's installed and in your PATH.")
    sys.exit(1)
    
# Time to wait between runs (in seconds)
WAIT_TIME = 5


def kill_existing_processes(process_name):
    try:
        subprocess.run(["pkill", "-f", process_name])
        time.sleep(2)  # Give time for the process to fully terminate
    except subprocess.CalledProcessError:
        print(f"No existing {process_name} processes found.")


def get_window_position(window_name):
    if not WMCTRL_PATH:
        return None
    try:
        output = subprocess.check_output([WMCTRL_PATH, "-l", "-G"]).decode()
        for line in output.splitlines():
            if window_name in line:
                parts = line.split()
                return (parts[2], parts[3])  # X and Y coordinates
    except subprocess.CalledProcessError:
        pass
    return None


def set_window_position(window_name, x, y):
    try:
        subprocess.run(["wmctrl", "-r", window_name, "-e", f"0,{x},{y},-1,-1"])
    except subprocess.CalledProcessError:
        print("Failed to set window position")


def run_binary(file_path, window_name, x=None, y=None):
    process = subprocess.Popen([BINARY_PATH, file_path])
    time.sleep(2)  # Give the window more time to open
    if x is not None and y is not None:
        set_window_position(window_name, x, y)
    return process


def monitor_and_rerun(file_path, window_name, interval=5):
    x, y = None, None
    process = None
    try:
        while True:
            print(f"Running URDF viewer for {file_path}")
            kill_existing_processes("urdf-viz")
            process = run_binary(file_path, window_name, x, y)
            
            if x is None and y is None:
                time.sleep(3)  # Give more time for the first run
                position = get_window_position(window_name)
                if position is None and WMCTRL_PATH:
                    print("Couldn't get window position. Window may move on refresh.")
            
            
            time.sleep(interval)
            
            if process:
                process.terminate()
                process.wait(timeout=5)  # Wait up to 5 seconds for the process to terminate
                if process.poll() is None:
                    process.kill()  # Force kill if it doesn't terminate
            
            print("Restarting URDF viewer...")
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_urdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    window_name = os.path.basename(file_path)  # Use filename as window name
    
    print(f"Monitoring and refreshing URDF: {file_path}")
    print("Press Ctrl+C to stop the script.")

    monitor_and_rerun(file_path, window_name)