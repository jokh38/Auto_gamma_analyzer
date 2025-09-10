import subprocess
import os

script_dir = "C:\Users\breezing\OneDrive\2025 작업\Code_작성중\Auto_gamma_analyzer"
script_name = "get_physical_origin_values.py"
full_script_path = os.path.join(script_dir, script_name)

try:
    result = subprocess.run(["python", full_script_path], capture_output=True, text=True, check=True)
    print(result.stdout)
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error executing script: {e}")
    print(f"Stdout: {e.stdout}")
    print(f"Stderr: {e.stderr}")
