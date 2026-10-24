"""Explicit GPU renderer smoke check; run in the simulator environment on a free device."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='0')
    p.add_argument('--gpu-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).resolve().parent / 'eval_runtime/src'
    sys.path.insert(0,str(code))
    from eval_platform.resources import resolve_gpus,gpu_lock,check_gpu
    gpu = resolve_gpus([a.gpu])[0]
    os.environ['LINGBOT_GPU_LOCK'] = str(a.gpu_lock.absolute())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    with gpu_lock(a.output.parent,gpu):
        check_gpu(gpu)
        apps = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        assert not apps.strip(), 'Existing compute process; no smoke scene started'
        pci = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
        import numpy as np
        import sapien
        sapien.render.set_camera_shader_dir('rt')
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser('oidn')
        render = sapien.render.RenderSystem('cuda:0')
        device = render.device
        prop = lambda n:getattr(device,n)() if callable(getattr(device,n)) else getattr(device,n)
        normalize = lambda bus:':'.join(str(int(v,16)) for v in bus.lower().split(':')[:-1])+':'+bus.lower().split(':')[-1]
        assert normalize(prop('pci_string'))==normalize(pci) and prop('can_render')
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(),render])
        scene.add_ground(0)
        scene.set_ambient_light([.5,.5,.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[.1,.1,.1],color=[1,0,0,1])
        actor = builder.build_static(name='reproduction_smoke_cube')
        actor.set_pose(sapien.Pose([0,0,.1]))
        camera = scene.add_camera('smoke',256,256,1.0,.01,10)
        camera.set_pose(sapien.Pose([-.8,0,.2]))
        scene.update_render();camera.take_picture()
        image = camera.get_picture('Color')
        assert image.shape==(256,256,4) and np.isfinite(image).all() and np.ptp(image[:,:,:3])>.001
        receipt = dict(passed=True,physical_gpu_uuid=gpu,pci_bus_id=pci,
                       shader='rt',samples_per_pixel=32,max_path_depth=8,oidn=True,
                       actual_smoke_scene_rendered=True,shape=list(image.shape),
                       robot_task_or_policy_acceptance=False)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf8') as stream:
            stream.write(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
