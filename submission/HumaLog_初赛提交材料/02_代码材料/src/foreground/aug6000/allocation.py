"""Allocate the same marginal branch budget to trustworthy masks first.

No dataset/index is redrawn. Plain and whole-head-mask choices are unchanged.
Only the ordinary/rectangle split is conditioned on verified mask eligibility.
The eligibility estimate is frozen before formal training and checkpoint-bound.
"""
from dataclasses import replace
import numpy as np
from .images import dilate
from .policy import branch_probabilities,rng_for

ELIGIBILITY_ESTIMATE=917/2048
DISABLED_RECTANGLE_TASKS=('press_stapler',)
CONTRACT='audited_eligibility_stratified_visible_branch_v1'

def pair_eligible(foregrounds,raw):
    if foregrounds is None or len(foregrounds)!=2:return False
    try:masks=[f.validated(raw[0,t]) for t,f in enumerate(foregrounds)]
    except ValueError as exc:
        if 'SHA' in str(exc):raise
        return False
    union=dilate(masks[0]|masks[1]);union[:12]=True;union[-12:]=True
    union[:,:12]=True;union[:,-12:]=True
    return float(union.mean())<.8

def conditional_rates(step,estimate=ELIGIBILITY_ESTIMATE):
    if not 0<estimate<1:raise ValueError('Frozen measured eligible-pair rate required')
    probabilities=branch_probabilities(step)
    visible=float(probabilities[1]+probabilities[2])
    target=float(probabilities[2])/visible
    if target<=estimate:return target/estimate,0.
    return 1.,(target-estimate)/(1-estimate)

def allocate(plan,eligible):
    if type(eligible) is not bool:raise ValueError('Explicit checked pair eligibility required')
    good,bad=conditional_rates(plan.step)
    original=plan.branch
    if original in ('ordinary','protected_background_rectangle'):
        probability=good if eligible else bad
        selected=rng_for(plan.draw,'verified_pair_branch_allocation').random()<probability
        plan=replace(plan,branch='protected_background_rectangle' if selected else 'ordinary')
    return plan,dict(contract=CONTRACT,original_plan_branch=original,
        verified_pair_eligible=eligible,frozen_eligibility_estimate=ELIGIBILITY_ESTIMATE,
        eligible_conditional_rectangle_probability=good,
        ineligible_conditional_rectangle_probability=bad,
        no_data_index_resampling=True,original_plain_and_whole_mask_choice_retained=True)
