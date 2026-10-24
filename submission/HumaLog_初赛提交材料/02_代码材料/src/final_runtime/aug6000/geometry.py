"""Independent45percent marginal primary-view geometry, no foreground veto."""
from dataclasses import replace
import numpy as np
from .policy import BRANCHES,branch_probabilities,rng_for

CONTRACT='primary_affine_global45_scale09_11_translate_lt6percent_v1'
GLOBAL_PROBABILITY=.45

def geometry_plan(plan):
    probabilities=branch_probabilities(plan.step)
    visible=float(probabilities[1]+probabilities[2])
    probability=min(1.,GLOBAL_PROBABILITY/visible)
    r=rng_for(plan.draw,'independent_geometry45_v1')
    selected=plan.branch in BRANCHES[1:3] and r.random()<probability
    scale=float(r.uniform(.9,1.1)) if selected else 1.
    # At256px, rounding +/- .06*256 yields at most15px, strictly below6percent.
    tx,ty=np.rint(r.uniform(-.06,.06,2)*256).astype(int) if selected else (0,0)
    return replace(plan,scale=scale,tx=int(tx),ty=int(ty))
