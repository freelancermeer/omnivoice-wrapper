r"""Does `reserved` climb across a run? Run this after any long batch.

    venv\Scripts\python tools\soak_vram.py 24 RVoiceover_3

A production batch reported idle VRAM going 3676 -> 9008 MB on an 8 GB card,
`free_mb` reaching 0, and every substantial request afterwards dying silently.
That is the shape this looks for.

It runs against the live server, so it measures the real path rather than a
fixture. The important part is reading `reserved` BEFORE the
cache release as well as after: a number taken only after `empty_cache()` is
blind to exactly the growth being hunted, which is why this went unnoticed.
"""
import json, sys, time
import requests

BASE = "http://127.0.0.1:8001"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
VOICE = sys.argv[2] if len(sys.argv) > 2 else "RVoiceover_3"

TEXT = ("The committee reconvened after lunch and the questions changed shape "
        "entirely. What had been an argument about paperwork became an argument "
        "about who knew what, and when. The chair asked for the ledger twice.")


def vram():
    return requests.get(BASE + "/api/health", timeout=60).json()["vram"]


base = vram()
print(f"start   alloc {base['allocated_mb']:7.0f}  reserved {base['reserved_mb']:7.0f}"
      f"  frag {base['fragmentation_mb']:6.0f}  free {base['free_mb']:7.0f}")

worst_free = base["free_mb"]
for i in range(1, N + 1):
    r = requests.post(BASE + "/api/tts",
                      data={"text": TEXT, "voice_id": VOICE, "steps": "16",
                            "format": "wav", "project": f"soak{i}"},
                      timeout=1800)
    if r.status_code != 200:
        print(f"  [{i:>3}] HTTP {r.status_code} {str(r.text)[:120]}")
        continue
    if i % 4 == 0 or i == N:
        v = vram()
        worst_free = min(worst_free, v["free_mb"])
        print(f"  [{i:>3}] alloc {v['allocated_mb']:7.0f}  "
              f"reserved {v['reserved_mb']:7.0f}  frag {v['fragmentation_mb']:6.0f}"
              f"  free {v['free_mb']:7.0f}   rtf {r.headers.get('X-RTF')}")

end = vram()
grew = end["reserved_mb"] - base["reserved_mb"]
print()
print(f"reserved  {base['reserved_mb']:.0f} -> {end['reserved_mb']:.0f} MB "
      f"({grew:+.0f} MB over {N} generations)")
print(f"free      {base['free_mb']:.0f} -> {end['free_mb']:.0f} MB "
      f"(lowest seen {worst_free:.0f})")
print(f"spilled to shared memory: "
      f"{requests.get(BASE + '/api/metrics', timeout=60).json().get('spilled_to_shared')}")
print()
print("LEAK" if grew > 200 else "no meaningful growth",
      f"— threshold is 200 MB; measured {grew:+.0f} MB")
