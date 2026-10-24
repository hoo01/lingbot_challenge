"""Actual installed official sampler and parent loader snapshot, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import copy,itertools,json
import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from common import ROOT,PARENT,atomic,sha
from stage_sampler import StageSampler,build_loader,FRAMES,CONTRACT
from aug6000.policy import FIRST_DRAW,GLOBAL_BATCH

class Examples:
    def __init__(self,parent_state,allow_parent=True):self.parent=parent_state;self.allow=allow_parent
    def __len__(self):return FRAMES
    def __getitem__(self,key):return dict(index=key[0],draw=key[1],epoch=key[2])
    def state_dict(self):return dict(contract='synthetic_boundary_test_v1')
    def load_state_dict(self,value):
        assert value==self.state_dict() or (self.allow and value==self.parent)

def expected(parent,index_count,rank):
    official=StatefulDistributedSampler(range(FRAMES),num_replicas=4,rank=rank,shuffle=True,seed=42)
    official.set_epoch(parent['epoch']);official.load_state_dict(parent['official'])
    return list(itertools.islice(official,index_count))

def main():
    torch.set_num_threads(1)
    out=ROOT/'artifacts/batch192-boundary-cpu.json';assert not out.exists()
    all_draws=[];proofs=[]
    checkpoint=PARENT/'runs/formal_8000_to10000/checkpoint_sft_010000'
    for rank in range(4):
        path=checkpoint/f'data_rank{rank}.pt'
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['step']==10000
        snapshot=value['loader']['_snapshot'];main=snapshot['_main_snapshot']
        state=main['_sampler_iter_state']['sampler_state']
        assert state['official']['yielded']==128000 and state['epoch']==0
        parent_dataset=snapshot['_worker_snapshots']['worker_0']['dataset_state']
        dataset=Examples(parent_dataset)
        loader,sampler,generator=build_loader(dataset,rank,allow_parent_boundary=True)
        loader.load_state_dict(copy.deepcopy(value['loader']));generator.set_state(value['generator'])
        iterator=iter(loader);first=[next(iterator) for _ in range(3)]
        indices=torch.cat([b['index'] for b in first]).tolist()
        draws=torch.cat([b['draw'] for b in first]).tolist();all_draws.extend(draws)
        assert indices==expected(state,48,rank)
        saved=copy.deepcopy(loader.state_dict())
        forward=[next(iterator) for _ in range(6)]
        restored,s2,g2=build_loader(Examples(parent_dataset,False),rank)
        restored.load_state_dict(saved);g2.set_state(generator.get_state())
        replay=iter(restored)
        for a in forward:
            b=next(replay)
            assert all(torch.equal(a[k],b[k]) for k in a)
        # Check epoch tail: remaining9435 ->9408, drop27; stage counter continuous.
        bare=StageSampler(range(FRAMES),rank,allow_parent_boundary=True);bare.load_state_dict(state)
        tail=list(bare);assert len(tail)==9408 and bare.stage_local_yielded==9408
        assert [x[0] for x in tail]==expected(state,9408,rank)
        bare.set_epoch(1);head=list(itertools.islice(bare,48))
        assert min(x[1] for x in head)>=FIRST_DRAW+196*GLOBAL_BATCH
        assert bare.stage_local_yielded==9456
        proofs.append(dict(rank=rank,parent_data_sha256=sha(path),parent_official_committed=128000,
            first48_actual_indices_equal_official=True,prefetch_resume_six_microbatches_exact=True,
            epoch_tail_samples=9408,dropped_epoch_tail=27,new_stage_draw_counter_continuous=True))
        iterator._shutdown_workers();replay._shutdown_workers()
    assert sorted(all_draws)==list(range(FIRST_DRAW,FIRST_DRAW+GLOBAL_BATCH))
    atomic(out,dict(passed=True,rank_proofs=proofs,first192_unique_exact=True,
        actual_installed_official_sampler=True,actual_parent_loader_snapshot=True,
        model_or_optimizer_weights_read=False,contract=CONTRACT))
    print(json.dumps(dict(passed=True,ranks=4,first192_unique_exact=True)))

if __name__=='__main__':main()
