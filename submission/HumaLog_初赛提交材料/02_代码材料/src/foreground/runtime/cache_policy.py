"""Conservative automatic eligibility under a separately audited detector policy."""
import re
import numpy as np
from scipy.ndimage import maximum_filter

from aug6000.grounding import expanded_box_mask

ALIASES={
 'robot':('robot','gripper'), 'cube':('cube','block'), 'bottle':('bottle',),
 'hammer':('hammer',), 'clock':('clock',), 'bell':('bell',),
 'trash':('trash','bin'), 'roller':('rolling pin','roller'), 'microphone':('microphone',),
 'mug':('mug','cup'), 'rack':('rack',), 'pot':('pot',), 'can':('can',),
 'pad':('pad','marker','mat','coaster'), 'cards':('card',), 'stapler':('stapler',),
 'laptop':('laptop',), 'microwave':('microwave',), 'basket':('basket',),
 'bread':('bread',), 'skillet':('skillet','pan'), 'burger':('hamburger','burger'),
 'fries':('fries',), 'tray':('tray',), 'box':('box',), 'bowl':('bowl',),
 'plate':('plate',), 'shoe':('shoe',), 'cup':('cup',), 'fan':('fan',),
 'mouse':('mouse',), 'scale':('scale',), 'stand':('stand','holder'),
 'phone':('smartphone','phone'), 'cabinet':('cabinet',), 'sign':('sign',),
 'scanner':('scanner',), 'stamp':('stamp','seal'), 'switch':('switch',),
 'soap':('soap',), 'toycar':('toy car','car'), 'remote':('remote',), 'container':('container',),
}

def role(phrase):
    p=phrase.lower()
    # More specific compounds precede their constituent words.
    for tokens,key in ((('robot','gripper'),'robot'),(('shoe box','storage box'),'box'),
        (('weighing scale',),'scale'),(('phone holder','display stand'),'stand'),
        (('cabinet',),'cabinet'),(('microwave',),'microwave'),(('alarm clock','clock button'),'clock'),
        (('rolling pin',),'roller'),(('square pad','marker','coaster'),'pad'),
        (('rubik',),'cube'),(('tea box','coffee box'),'box'),(('playing cards',),'cards'),
        (('dinner plate',),'plate'),(('french fries',),'fries'),(('toy car',),'toycar')):
        if any(t in p for t in tokens):return key
    for key,aliases in ALIASES.items():
        if any(re.search(r'\b'+re.escape(a)+r'(?:s)?\b',p) for a in aliases):return key
    raise ValueError('Unknown detector-only object role: '+phrase)

def detected_roles(detections):
    found=set()
    for d in detections:
        label=d['label'].lower()
        for key,aliases in ALIASES.items():
            if any(re.search(r'\b'+re.escape(a)+r'(?:s)?\b',label) for a in aliases):found.add(key)
    return found

def select(frame,phrases,*,disabled=False):
    selected=[d for d in frame['detections'] if d['score']>=.25]
    required=sorted({role(p) for p in phrases})
    found=detected_roles(selected)
    mask,_=expanded_box_mask([d['box'] for d in selected])
    protected=maximum_filter(mask,size=15,mode='constant',cval=False)
    protected[:12,:]=True;protected[-12:,:]=True
    protected[:,:12]=True;protected[:,-12:]=True
    missing=sorted(set(required)-found)
    fraction=float(protected.mean())
    eligible=not disabled and not missing and bool(mask.any()) and fraction<.8
    reason='task_disabled_after_visual_audit' if disabled else (
        'missing_object_roles' if missing else 'too_much_protection' if fraction>=.8 else None)
    return mask,dict(rgb_sha256=frame['rgb_sha256'],reviewed=False,
        policy_eligible=eligible,required_labels=required,covered_labels=sorted(found),
        missing_roles=missing,protected_fraction=fraction,fallback=reason,
        entity_boxes=[d['box'] for d in selected],
        trust_basis='audited_policy_plus_per_frame_checks; not individual manual review')
