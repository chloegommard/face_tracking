import re
from collections import defaultdict

LOG_FILE = "performance.log"

# Stockage des valeurs par étape
times = defaultdict(list)

pattern = re.compile(r"\| (.+?): ([\d.]+) ms")

with open(LOG_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            step = match.group(1).strip()
            value = float(match.group(2))
            times[step].append(value)


print("\n===== MOYENNES =====\n")

total_frames = 0

for step, values in times.items():
    avg = sum(values) / len(values)
    print(f"{step:<30} {avg:8.3f} ms  ({len(values)} mesures)")

    if step == "FRAME TOTAL":
        total_frames = len(values)


if total_frames:
    print("\nFrames analysées :", total_frames)
    print(f"FPS théorique moyen : {1000 / (sum(times['FRAME TOTAL']) / len(times['FRAME TOTAL'])):.1f}")