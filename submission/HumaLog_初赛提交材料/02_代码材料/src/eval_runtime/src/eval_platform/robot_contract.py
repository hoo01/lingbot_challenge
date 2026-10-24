"""Optional model-specific action configuration; legacy absolute models unchanged."""
import os
import hashlib
import json
from pathlib import Path
import yaml

class BoundRobotConfigMixin:
    def reset(self,robo_name,path_to_pi_model=None):
        if path_to_pi_model is not None:
            self.vla=self.load_vla(path_to_pi_model).cuda().eval()
            if self.use_bf16:self.vla=self.vla.bfloat16()
            else:self.vla.model.float()
        if not getattr(self.data_config,'robot_config_path',None):
            return super().reset(robo_name,None)
        from lingbotvla.data.vla_data.utils import FeatureTransform
        path=Path(self.data_config.robot_config_path)
        proof=json.loads(Path(self.robot_norm_path).with_suffix('.provenance.json').read_text())
        scope=json.loads((Path(os.environ['LINGBOT_ROOT'])/'artifacts/eval2000-scope.json').read_text())
        assert proof['robot_config_sha256']=='1ef96ff0d213561174813345f8674fc693f4831cde144e00e5b9781d7147876b'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==scope['robot_sha256']=='f4a1de74470ae9d7353bbfe7bdd173305bf391cecd05f4dd7f5276968e426c5e'
        assert hashlib.sha256(Path(self.robot_norm_path).read_bytes()).hexdigest()==proof['norm_sha256']
        assert robo_name=='robotwin'
        self.robot_config=yaml.safe_load(path.read_text())
        self.global_step=0;self.last_action_chunk=None;self.last_normalized_action_chunk=None
        self.vla.feature_transform=FeatureTransform(str(path),self.data_config,self.config,self.processor,
            chunk_size=self.config.chunk_size,norm_stats_path=self.robot_norm_path)
        self.action_key=self.vla.feature_transform.org_features['actions']
        transform = self.vla.feature_transform
        self.robot_action_contract = dict(robot_config_path=str(path),
            robot_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            normalization_sha256=proof['norm_sha256'], norm_type=self.data_config.norm_type,
            action_subtract_state=transform.action_subtract_state,
            action_relative_type=transform.action_relative_type,
            action_features=transform.actions, state_features=transform.states,
            chunk_size=self.config.chunk_size,
            statistic_shapes={key:{name:list(value.shape) for name,value in stats.items()
                if hasattr(value,'shape')} for key,stats in transform.normalizer.norm_stats.items()})
