"""Continue the actual official cursor; tag new batch192 draws independently."""
from itertools import islice
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from aug6000.policy import draw_for_local_position,GLOBAL_BATCH,WORLD,MICRO,UPDATES

FRAMES=549737
CONTRACT='official_cursor_aug6000_global192_v1'

class StageSampler(Sampler):
    def __init__(self,dataset,rank,*,allow_parent_boundary=False):
        assert len(dataset)==FRAMES and 0<=rank<WORLD
        self.dataset,self.rank=dataset,int(rank)
        self.allow_parent_boundary=allow_parent_boundary
        self.official=StatefulDistributedSampler(dataset,num_replicas=WORLD,rank=rank,shuffle=True,seed=42)
        self.epoch=0;self.stage_local_yielded=0

    def __iter__(self):
        start=self.official.next_yielded or 0
        remaining=max(0,((len(self.official)-start)//48)*48)
        remaining=min(remaining,UPDATES*48-self.stage_local_yielded)
        for index in islice(self.official,remaining):
            position=self.stage_local_yielded
            self.stage_local_yielded+=1
            yield int(index),int(draw_for_local_position(position,self.rank)),self.epoch

    def __len__(self):return (len(self.official)//48)*48

    def set_epoch(self,epoch):
        self.epoch=int(epoch)
        self.official.set_epoch(self.epoch)
        self.official.load_state_dict({'yielded':0})

    def state_dict(self):
        return dict(contract=CONTRACT,epoch=self.epoch,rank=self.rank,world=WORLD,
                    official=self.official.state_dict(),stage_local_yielded=self.stage_local_yielded)

    def load_state_dict(self,state):
        assert (state['rank'],state['world'])==(self.rank,WORLD)
        self.set_epoch(state['epoch']);self.official.load_state_dict(state['official'])
        if state.get('contract')==CONTRACT:
            position=state['stage_local_yielded'];assert 0<=position<=UPDATES*48
            self.stage_local_yielded=int(position)
        else:
            assert self.allow_parent_boundary and set(state)=={'epoch','rank','world','official'}
            self.stage_local_yielded=0

def build_loader(dataset,rank,workers=2,*,allow_parent_boundary=False):
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    sampler=StageSampler(dataset,rank,allow_parent_boundary=allow_parent_boundary)
    generator=torch.Generator().manual_seed(42+rank)
    loader=StatefulDataLoader(dataset,batch_size=MICRO,sampler=sampler,num_workers=workers,
        multiprocessing_context='spawn',persistent_workers=False,prefetch_factor=2,
        drop_last=True,pin_memory=True,generator=generator)
    return loader,sampler,generator

def next_batch(loader,sampler,iterator):
    try:return next(iterator),iterator
    except StopIteration:
        sampler.set_epoch(sampler.epoch+1)
        iterator=iter(loader)
        return next(iterator),iterator
