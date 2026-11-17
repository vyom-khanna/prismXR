"""Build (url, output path, md5) for the manifest's EPIC videos, replicating epic_downloader.py's URL logic,
so they can be fetched with a RESUMABLE downloader (curl -C -). The official script restarts files from
zero whenever the Bristol server resets a connection.

    py -3.13 tools/epic_urls.py > C:/prism_data/epic_urls.tsv
"""
import csv
import os

DL = r"C:/prism_ext/epic-kitchens-download-scripts"
OUT = r"C:/prism_data/epic/EPIC-KITCHENS"
BASE55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
BASE100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")

e55 = {r["video_id"]: r["split"] for r in csv.DictReader(open(os.path.join(DL, "data", "epic_55_splits.csv")))}
md5 = {}
for r in csv.DictReader(open(os.path.join(DL, "data", "md5.csv"))):
    md5[(r["version"], r["file_remote_path"])] = r["md5"]
errata = {}
p = os.path.join(DL, "data", "errata.csv")
if os.path.exists(p):
    for r in csv.DictReader(open(p)):
        errata[r[list(r.keys())[0]]] = r[list(r.keys())[1]]

with open(MANIFEST) as f:
    vids = [r["video_id"] for r in csv.DictReader(f) if r["dataset"] == "epic_kitchens_100"]
for v in vids:
    part = v.split("_")[0]
    if v in e55:
        remote = f"videos/{e55[v]}/{part}/{v}.MP4"
        url, ver = f"{BASE55}/{remote}", "55"
    else:
        remote = f"{part}/videos/{v}.MP4"
        url, ver = f"{BASE100}/{remote}", "100"
    if remote in errata:
        url, ver = errata[remote], "errata"
    print("\t".join([v, url, f"{OUT}/{part}/videos/{v}.MP4", md5.get((ver, remote), "")]))
