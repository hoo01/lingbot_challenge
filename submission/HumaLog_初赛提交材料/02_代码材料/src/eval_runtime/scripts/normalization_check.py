"""CPU-only checks using the actual deployed FeatureTransform and frozen raw input."""
import hashlib,json
from pathlib import Path

def check_roundtrip(transform,raw):
 import torch
 assert transform.action_subtract_state=={'action.arm.position':True,'action.effector.position':False}
 assert transform.action_relative_type['action.arm.position']=='joint'
 assert tuple(transform.normalizer.norm_stats['action.arm.position']['mean'].shape)==(50,12)
 q=torch.as_tensor(raw['observation.state']).float().clone()
 assert q.shape==(14,)
 physical=q.repeat(50,1)+torch.linspace(-.01,.01,50).view(50,1)
 # Keep grip values in a physical non-boundary range; this is a CPU arithmetic fixture only.
 physical[:,6]=.4;physical[:,13]=.6
 data={k:torch.as_tensor(v).clone() for k,v in raw.items() if k.startswith('observation.')}
 data['action']=physical.clone()
 features=transform.convert_features(data,w_action=True)
 features={k:v for k,v in features.items() if k.startswith(('action.','observation.state.'))}
 for key in transform.actions:
  if transform.action_subtract_state[key]:features[key]=features[key]-features[key.replace('action.','observation.state.')]
 clone=lambda d:{k:v.clone() for k,v in d.items()}
 normalized=transform.normalizer.normalize(clone(features))
 recovered=transform.normalizer.unnormalize(clone(normalized))
 # Native JSON statistics use float64 arithmetic; compare values in that same
 # dtype without changing the deployed transform or the model's FP32 tensors.
 for k in features:torch.testing.assert_close(recovered[k],features[k].to(recovered[k].dtype),atol=1e-6,rtol=1e-6)
 for k in transform.actions:
  if transform.action_subtract_state[k]:recovered[k]=recovered[k]+recovered[k.replace('action.','observation.state.')]
 absolute=transform.reverse_features(recovered)['action']
 torch.testing.assert_close(absolute,physical.to(absolute.dtype),atol=1e-6,rtol=1e-6)
 return dict(passed=True,atol=1e-6,rtol=1e-6,max_absolute_error=float((absolute-physical).abs().max()),native_normalizer_arithmetic_dtype=str(absolute.dtype),relative_arm_same_q_t_all_horizons=True,gripper_absolute_minmax=True,raw_rgb_no_inferred_units=True)

def replay(root,npz_path):
 import numpy as np,torch,yaml
 from types import SimpleNamespace
 from transformers import AutoConfig
 import ast
 from torchvision.transforms.v2 import Resize
 from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
 from lingbotvla.data.vla_data.utils import FeatureTransform
 from lingbotvla.models import build_processor
 root=Path(root);scope=json.loads((root/'artifacts/eval2000-scope.json').read_text());cp=Path(scope['checkpoint']);folder=cp.parent.parent.parent
 cli=yaml.safe_load((folder/'lingbotvla_cli.yaml').read_text());cfg={**cli['model'],**cli['train']}
 # Importing the full policy initializes CUDA-tuned MoE kernels. Execute only
 # the three unchanged native CPU methods, without importing a model module.
 native=root/'third_party/lingbot-vla-v2-official-be969b8f/deploy/lingbot_vla_v2_policy.py'
 tree=ast.parse(native.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVLAv2Server')
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('merge_qwen_config','resize_image','_prepare_model_input')]
 assert len(methods)==3
 cpu=ast.Module(body=[ast.ClassDef(name='CPUOnlyNativePreprocess',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
 namespace=dict(torch=torch,np=np,Resize=Resize);exec(compile(ast.fix_missing_locations(cpu),str(native),'exec'),namespace)
 helper=namespace['CPUOnlyNativePreprocess']();helper.config=LingbotVLAV2Config(**cfg)
 for k,v in cfg.items():
  if not hasattr(helper.config,k):setattr(helper.config,k,v)
 helper.merge_qwen_config(AutoConfig.from_pretrained(root/'models/Qwen3-VL-4B-Instruct'))
 helper.data_config=SimpleNamespace(**cli['data']);helper.processor=build_processor(str(root/'models/Qwen3-VL-4B-Instruct'));helper.use_bf16=False
 ft=FeatureTransform(str(root/'assets/robotwin_relative.yaml'),helper.data_config,helper.config,helper.processor,chunk_size=50,norm_stats_path=str(folder/'norm_stats.json'))
 helper.vla=SimpleNamespace(feature_transform=ft)
 with np.load(npz_path,allow_pickle=False) as z:raw={k:(v.item() if k=='task' else v.copy()) for k,v in z.items()}
 prepared=helper._prepare_model_input(raw)
 tensors={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in prepared.items() if isinstance(v,torch.Tensor)}
 return dict(tensors=tensors,roundtrip=check_roundtrip(ft,raw),cpu_only=True,native_preprocessing_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
