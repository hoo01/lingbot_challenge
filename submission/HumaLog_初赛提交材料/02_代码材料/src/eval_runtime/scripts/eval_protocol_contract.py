"""Fail closed when a server does not match the requested inference profile."""
def validate_profile(metadata, profile):
    if profile=='aug16000_h50_execute25_no_video_v1':
        expected={'inference_profile':profile,'dtype':'fp32','execute':25,'prediction_horizon':50,
                  'execution_protocol':'H50_first25_reobserve_v1',
                  'moe_backend':'official','vision_device':'cuda','embedding_device':'cuda',
                  'flow_noise_generation':'upstream_native','sampling_seed_at_process_start':42,
                  'sampling_seed_per_episode':None,'use_compile':True,'image_size':256,
                  'num_steps':10,'vit_attn_implementation':'flash_attention_2',
                  'precompute_grid_thw':True,'matmul_precision':'high'}
        for key,value in expected.items():
            if key not in metadata or metadata[key]!=value:
                raise ValueError(f'Official inference mismatch: {key} expected {value!r}, got {metadata.get(key)!r}')
    elif profile=='research4_paired_v1':
        expected={'inference_profile':profile,'dtype':'fp32','moe_backend':'torch_reference',
                  'vision_device':'cpu','embedding_device':'cpu','image_size':256,'num_steps':10,
                  'use_compile':False,'sampling_seed_per_episode':42,
                  'flow_noise_generation':'fp32_then_cast_per_inference','matmul_precision':'highest'}
        for key,value in expected.items():
            if metadata.get(key)!=value:
                raise ValueError(f'Research inference mismatch: {key}: {metadata.get(key)} vs {value}')
        if metadata.get('execute') not in (10,25,50):
            raise ValueError('Unsupported research execution length')
    elif profile=='paired_legacy':
        assert metadata.get('sampling_seed_per_episode')==42,metadata
        assert metadata.get('flow_noise_generation')=='fp32_then_cast_per_inference',metadata
    else:
        raise ValueError(profile)
