"""DEV: run the submission's EPIC stage (prism.pipeline.epic_actions + select_coverage) on cached SlowFast
features for all 138 val videos and report fold / all-138 mAP and coverage. Same code path as run.sh.

    C:/prism_env/Scripts/python.exe tools/eval_pipeline_epic.py
"""
import numpy as np

import epic_dev as E
from prism import pipeline as PL, select_coverage as SC

acts, segs = {}, {}
for i, v in enumerate(E.EPIC):
    with np.load(E.feat_file(v)) as z:
        F = {k: z[k] for k in z.files}
    a, dens, sel = PL.epic_actions(F, E.MAN[v][0], {})
    acts[v] = a
    segs[v] = SC.select(sel, E.MAN[v][0], SC.CovCfg(), density=dens)
m, c = E.report("submission EPIC stage (current code)", acts, segs=segs)
print(f"all-138 pair mAP {E.pair_map(acts):.2f}   coverage {E.coverage(segs):.2f}")
