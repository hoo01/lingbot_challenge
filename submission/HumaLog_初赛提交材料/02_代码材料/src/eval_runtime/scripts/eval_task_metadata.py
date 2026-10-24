"""Official task language fields and initial scoring state, with no expert rollout.

Derived from RoboTwin commit 13c3c47ff4312dd62484bcd51be034af55c062d1.
Only language substitutions are passed to the policy. Initial scoring attributes
otherwise assigned by play_once are reproduced without moving the simulation.
"""

def scene_language_parameters(task, e):
    def arm(actor):
        return 'left' if actor.get_pose().p[0] < 0 else 'right'
    def opposite(value):
        return 'right' if str(value) == 'left' else 'left'
    def fields(*objects, a=None, b=None, c=None):
        result = {'{' + chr(65+i) + '}': x for i, x in enumerate(objects)}
        result.update({'{' + k + '}': str(v) for k, v in dict(a=a,b=b,c=c).items() if v is not None})
        return result
    if task == 'adjust_bottle':
        return fields(f'001_bottle/base{e.model_id}', a='right' if e.qpose_tag == 1 else 'left')
    if task == 'beat_block_hammer':
        return fields('020_hammer/base0', a='left' if e.block.get_functional_point(0,'pose').p[0] < 0 else 'right')
    if task in ('blocks_ranking_rgb','blocks_ranking_size','stack_blocks_two','stack_blocks_three'):
        objects = ['large block','medium block','small block'] if task.endswith('size') else ['red block','green block','blue block']
        count = 2 if task.endswith('two') else 3
        return fields(*objects[:count], **{k:arm(getattr(e,f'block{i+1}')) for i,k in enumerate('abc'[:count])})
    if task == 'click_alarmclock': return fields(f'046_alarm-clock/base{e.alarmclock_id}', a=arm(e.alarm))
    if task == 'click_bell': return fields(f'050_bell/base{e.bell_id}', a=arm(e.bell))
    if task == 'dump_bin_bigbin': return fields(f'063_tabletrashbin/base{e.deskbin_id}')
    if task == 'grab_roller': return fields(f'102_roller/base{e.model_id}')
    if task == 'handover_block': return {}
    if task == 'handover_mic': return fields(f'018_microphone/base{e.microphone_id}', a=arm(e.microphone), b=opposite(arm(e.microphone)))
    if task == 'hanging_mug': return fields(f'039_mug/base{e.mug_id}', '040_rack/base0')
    if task == 'lift_pot': return fields(f'{e.model_name}/base{e.model_id}')
    if task == 'move_can_pot': return fields(f'060_kitchenpot/base{e.pot_id}', f'105_sauce-can/base{e.can_id}', a=e.arm_tag)
    if task == 'move_pillbottle_pad': return fields(f'080_pillbottle/base{e.pillbottle_id}', a=arm(e.pillbottle))
    if task == 'move_playingcard_away': return fields(f'081_playingcards/base{e.playingcards_id}', a=arm(e.playingcards))
    if task == 'move_stapler_pad': return fields(f'048_stapler/base{e.stapler_id}', e.color_name, a=arm(e.stapler))
    if task == 'open_laptop':
        from envs.utils import ArmTag, get_face_prod
        e.arm_tag = ArmTag('left' if get_face_prod(e.laptop.get_pose().q,[1,0,0],[1,0,0]) > 0 else 'right')
        return fields(f'{e.model_name}/base{e.model_id}', a=e.arm_tag)
    if task == 'open_microwave': return fields(f'{e.model_name}/base{e.model_id}', a='left')
    if task == 'pick_diverse_bottles': return fields(f'001_bottle/base{e.bottle1_id}', f'001_bottle/base{e.bottle2_id}')
    if task == 'pick_dual_bottles': return fields('001_bottle/base13','001_bottle/base16')
    if task in ('place_a2b_left','place_a2b_right'):
        return fields(f'{e.selected_modelname_A}/base{e.selected_model_id_A}', f'{e.selected_modelname_B}/base{e.selected_model_id_B}', a=arm(e.object))
    if task == 'place_bread_basket':
        if not e.bread: raise RuntimeError('Official scene contains no bread')
        a = arm(e.bread[0]) if len(e.bread) == 1 or e.bread[0].get_pose().p[0]*e.bread[1].get_pose().p[0] > 0 else 'dual'
        return fields(f'076_breadbasket/base{e.basket_id}', *[f'075_bread/base{x}' for x in e.bread_id], a=a)
    if task == 'place_bread_skillet': return fields(f'106_skillet/base{e.skillet_id}', f'075_bread/base{e.bread_id}', a=arm(e.skillet))
    if task == 'place_burger_fries': return fields(f'006_hamburg/base{e.object1_id}', f'008_tray/base{e.tray_id}', f'005_french-fries/base{e.object2_id}')
    if task == 'place_can_basket': return fields(f'{e.can_name}/base{e.can_id}', f'{e.basket_name}/base{e.basket_id}', a=e.arm_tag)
    if task == 'place_cans_plasticbox': return fields(f'071_can/base{e.object1_id}', f'062_plasticbox/base{e.plasticbox_id}', f'071_can/base{e.object2_id}')
    if task == 'place_container_plate': return fields(f'003_plate/base{e.plate_id}', f'{e.actor_name}/base{e.container_id}', a=arm(e.container))
    if task == 'place_dual_shoes': return fields(f'041_shoe/base{e.shoe_id}', '007_shoe-box/base0')
    if task == 'place_empty_cup': return fields('021_cup/base0','019_coaster/base0')
    if task == 'place_fan': return fields(f'099_fan/base{e.fan_id}', e.color_name, a=arm(e.fan))
    if task == 'place_mouse_pad': return fields(f'047_mouse/base{e.mouse_id}', e.color_name, a=arm(e.mouse))
    if task == 'place_object_basket': return fields(f'{e.object_name}/base{e.object_id}', f'{e.basket_name}/base{e.basket_id}', a=e.arm_tag,b=opposite(e.arm_tag))
    if task == 'place_object_scale':
        from envs.utils import ArmTag
        e.arm_tag = ArmTag(arm(e.object))
        return fields(f'072_electronicscale/base{e.scale_id}', f'{e.selected_modelname}/base{e.selected_model_id}', a=e.arm_tag)
    if task == 'place_object_stand': return fields(f'{e.selected_modelname}/base{e.selected_model_id}', f'074_displaystand/base{e.displaystand_id}',a=arm(e.object))
    if task == 'place_phone_stand': return fields(f'077_phone/base{e.phone_id}', f'078_phonestand/base{e.stand_id}',a=arm(e.phone))
    if task == 'place_shoe': return fields(f'041_shoe/base{e.shoe_id}',a=arm(e.shoe))
    if task == 'press_stapler': return fields(f'048_stapler/base{e.stapler_id}',a=arm(e.stapler))
    if task == 'put_bottles_dustbin': return fields(*[f'114_bottle/base{x}' for x in e.bottle_id], '011_dustbin/base0')
    if task == 'put_object_cabinet':
        from envs.utils import ArmTag
        e.arm_tag = ArmTag(arm(e.object)); e.origin_z = e.object.get_pose().p[2]
        return fields(f'{e.selected_modelname}/base{e.selected_model_id}','036_cabinet/base0',a=e.arm_tag,b=opposite(e.arm_tag))
    if task == 'rotate_qrcode': return fields(f'070_paymentsign/base{e.model_id}',a=arm(e.qrcode))
    if task == 'scan_object': return fields(f'112_tea-box/base{e.object_id}',f'024_scanner/base{e.scanner_id}',a=opposite(arm(e.scanner)),b=arm(e.scanner))
    if task in ('shake_bottle','shake_bottle_horizontally'): return fields(f'001_bottle/base{e.bottle_id}',a=arm(e.bottle))
    if task == 'stack_bowls_three': return fields('002_bowl/base3')
    if task == 'stack_bowls_two': return fields('002_bowl/base3','002_bowl/base3',a=arm(e.bowl1),b=arm(e.bowl2))
    if task == 'stamp_seal': return fields(f'100_seal/base{e.seal_id}',e.color_name,a=arm(e.seal))
    if task == 'turn_switch': return fields(f'056_switch/base{e.model_id}',a='right' if -e.switch.get_pose().to_transformation_matrix()[0,0] > 0 else 'left')
    raise ValueError(f'No audited language metadata for {task}')
