import subprocess
import sys
import pathlib

skill_path = pathlib.Path(sys.argv[1])
content = skill_path.read_text(encoding="utf-8")

result = subprocess.run(
    ["claude", "-p", content],
    shell=True
)
sys.exit(result.returncode)
