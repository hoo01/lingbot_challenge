"""Detector-only object names from the original Clean task/instruction fields."""
import re

BASE={
 'adjust_bottle':'bottle', 'beat_block_hammer':'hammer. wooden block',
 'blocks_ranking_rgb':'red cube. green cube. blue cube',
 'blocks_ranking_size':'colored cubes', 'click_alarmclock':'alarm clock. clock button',
 'click_bell':'desk bell', 'dump_bin_bigbin':'small trash bin. large trash bin',
 'grab_roller':'wooden rolling pin', 'handover_block':'red cube. blue square pad',
 'handover_mic':'microphone', 'hanging_mug':'mug. mug rack', 'lift_pot':'cooking pot',
 'move_can_pot':'can. cooking pot', 'move_pillbottle_pad':'pill bottle. blue square pad',
 'move_playingcard_away':'deck of playing cards', 'move_stapler_pad':'stapler. square pad',
 'open_laptop':'laptop', 'open_microwave':'microwave. microwave door handle',
 'pick_diverse_bottles':'bottles', 'pick_dual_bottles':'bottles',
 'place_a2b_left':'', 'place_a2b_right':'',
 'place_bread_basket':'bread. basket', 'place_bread_skillet':'bread. skillet',
 'place_burger_fries':'hamburger. box of french fries. plastic tray',
 'place_can_basket':'can. basket', 'place_cans_plasticbox':'soda cans. plastic storage box',
 'place_container_plate':'bowl. container. dinner plate',
 'place_dual_shoes':'shoes. shoe box', 'place_empty_cup':'cup. coaster',
 'place_fan':'fan. square pad', 'place_mouse_pad':'computer mouse. square pad',
 'place_object_basket':'basket', 'place_object_scale':'electronic weighing scale',
 'place_object_stand':'display stand', 'place_phone_stand':'smartphone. phone holder',
 'place_shoe':'shoe. blue rectangular marker', 'press_stapler':'stapler',
 'put_bottles_dustbin':'bottles. trash bin', 'put_object_cabinet':'cabinet. cabinet door handle',
 'rotate_qrcode':'QR code payment sign', 'scan_object':'barcode scanner',
 'shake_bottle':'bottle', 'shake_bottle_horizontally':'bottle',
 'stack_blocks_three':'red cube. green cube. blue cube',
 'stack_blocks_two':'red cube. green cube', 'stack_bowls_three':'bowls',
 'stack_bowls_two':'bowls', 'stamp_seal':'rubber stamp. colored square marker',
 'turn_switch':'toggle switch',
}
NOUNS=(('rubik','rubik cube'),('stapler','stapler'),('mouse','computer mouse'),
 ('toycar','toy car'),('toy car','toy car'),('bread','bread'),('phone','smartphone'),
 ('playingcard','deck of playing cards'),('playing card','deck of playing cards'),
 ('soap','bar of soap'),('tea box','tea box'),('coffee box','coffee box'),
 ('bottle','bottle'),('bell','desk bell'),('woodenblock','wooden block'),
 ('wooden block','wooden block'),('block','block'),('microphone','microphone'),
 ('mug','mug'),('cup','cup'),('can','can'),('hamburg','hamburger'),
 ('remotecontrol','remote control'),('remote control','remote control'))
DIVERSE={'place_a2b_left','place_a2b_right','place_object_basket','place_object_scale',
         'place_object_stand','put_object_cabinet','scan_object'}

def phrases(record):
    task=record['task'];fixed=[p.strip() for p in BASE[task].split('.') if p.strip()]
    if task in DIVERSE:
        text=' '.join(record['instructions']).lower().replace('-',' ')
        extra=sorted({noun for token,noun in NOUNS if
            (re.search(r'\bcans?\b',text) if token=='can' else token in text)})
        if not extra:raise ValueError('No concrete original-instruction object names: '+task)
        fixed.extend(extra)
    return list(dict.fromkeys(['robot gripper','robot arm',*fixed]))

def prompt(record):return '. '.join(phrases(record))+'.'
