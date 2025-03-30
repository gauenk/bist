import time
import psutil

# Wait until the process is no longer running
pid = 1377414
while psutil.pid_exists(pid):
    time.sleep(1)  # Check every second

import subprocess
cmd = "/home/gauenk/.pyenv/versions/st_spix/bin/python ./dev/eval_ablation.py"
print(cmd)
output = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
print(output)
