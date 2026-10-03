#!/usr/bin/env python3
"""Render logs/dashboard.html from the training logs (rl/progress.py + rl/dashboard_template.html)."""
import json, os, subprocess, sys
RL = os.path.dirname(os.path.abspath(__file__))
out_json = os.path.join(RL, "..", "logs", "progress.json")
subprocess.run([sys.executable, os.path.join(RL, "progress.py"), "--json", out_json], check=True, stdout=subprocess.DEVNULL)
data = open(out_json).read()
html = open(os.path.join(RL, "dashboard_template.html")).read().replace("/*DATA*/null", data.replace("</", "<\\/"))
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RL, "..", "logs", "dashboard.html")
open(dst, "w").write(html)
print(dst)
