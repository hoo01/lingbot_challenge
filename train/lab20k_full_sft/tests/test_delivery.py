import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract


class InputTests(unittest.TestCase):
    def test_reject_manifest_from_a_different_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'delivery_manifest.json').write_text('{}')
            (root/'release.json').write_text('{"delivery_manifest_sha256":"'+('0'*64)+'"}')
            with patch.object(contract,'__file__',str(root/'contract.py')):
                with self.assertRaisesRegex(ValueError,'Manifest differs'):
                    contract.validate_bundle(root)

    def observation(self):
        return {**{k: np.zeros((8,9,3),dtype=np.uint8) for k in contract.CAMERAS},
                'observation.state': np.zeros(14), 'task':'click the bell'}

    def test_dark_uint8_rgb_keeps_its_original_units(self):
        obs = self.observation()
        contract.validate_observation(obs)
        self.assertEqual(obs[contract.CAMERAS[0]].dtype, np.uint8)

    def test_reject_pre_normalized_float_images(self):
        obs = self.observation()
        obs[contract.CAMERAS[0]] = obs[contract.CAMERAS[0]].astype(np.float32)/255
        with self.assertRaises(ValueError):
            contract.validate_observation(obs)

    def test_reject_wrong_or_nonfinite_joint_state(self):
        for state in (np.zeros(12), np.full(14,np.nan)):
            obs = self.observation()
            obs['observation.state'] = state
            with self.assertRaises(ValueError):
                contract.validate_observation(obs)

    def test_promote_before_loading_saved_fp32_values(self):
        events = []
        class Model:
            def float(self):
                events.append('float')
        class Base:
            def load_model_weights(self, path, strict=True):
                events.append(('load',path,strict))
        class Policy(contract.FP32BeforeLoadMixin,Base):
            vla = Model()
            delivery_audit = {}
        with patch.object(contract,'verify_loaded',return_value={'passed':True}):
            Policy().load_model_weights('checkpoint')
        self.assertEqual(events, ['float',('load','checkpoint',True)])


class UpstreamPrefixTests(unittest.TestCase):
    """Execute the real pinned upstream infer method with a deterministic fake model."""
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['UPSTREAM_DIR'])/'deploy/lingbot_vla_v2_policy.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        server = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVLAv2Server')
        infer = next(n for n in server.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
        namespace = {}
        exec(compile(ast.Module(body=[infer],type_ignores=[]),str(source),'exec'),namespace)
        cls.real_infer = staticmethod(namespace['infer'])

    def run_prefix(self, execute):
        class Fake:
            chunk_ret = True
            use_length = execute
            last_action_chunk = None
            last_normalized_action_chunk = None
            global_step = 0
            calls = 0
            predicted_horizon = 50
            def _infer_batch(self, observations):
                self.calls += 1
                return {'action':np.full((len(observations),50,14),self.calls,dtype=np.float32)}
        policy = Fake()
        for index in (1,2):
            actions = self.real_infer(policy, {'observation.state':np.full(14,index)})['action']
            self.assertEqual(actions.shape, (execute,14))
            self.assertTrue(np.all(actions == index))
        self.assertEqual(policy.calls,2)  # New observation => new H50, not stale cached tail.
        self.assertEqual(policy.predicted_horizon,50)

    def test_execute25_replans_each_request(self):
        self.run_prefix(25)

    def test_execute50_baseline_keeps_full_prediction(self):
        self.run_prefix(50)


if __name__ == '__main__':
    unittest.main()
