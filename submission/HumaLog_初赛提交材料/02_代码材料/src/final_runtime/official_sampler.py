"""Decorate the installed official sampler; do not change any frame indices."""
from itertools import islice
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from policy import WORLD,SEED,LOCAL_BATCH,GLOBAL_BATCH,UPDATES_PER_EPOCH,FRAMES,START,MICRO


class OfficialTaggedSampler(Sampler):
    def __init__(self,dataset,rank):
        assert len(dataset)==FRAMES and 0<=rank<WORLD
        # These are the official data_loader.py defaults. In particular the
        # sampler pads while the DataLoader drops its last incomplete batch.
        self.official=StatefulDistributedSampler(dataset,num_replicas=WORLD,rank=rank,
                                               shuffle=True,seed=SEED)
        self.rank=rank
        self.epoch=0

    def __iter__(self):
        start=self.official.next_yielded or 0
        remaining=max(0,UPDATES_PER_EPOCH*LOCAL_BATCH-start)
        for position,index in enumerate(islice(self.official,remaining),start=start):
            update=START+self.epoch*UPDATES_PER_EPOCH+position//LOCAL_BATCH
            micro,offset=divmod(position%LOCAL_BATCH,MICRO)
            draw=update*GLOBAL_BATCH+micro*WORLD*MICRO+self.rank*MICRO+offset
            yield int(index),int(draw),int(self.epoch)

    def __len__(self):return UPDATES_PER_EPOCH*LOCAL_BATCH

    def set_epoch(self,epoch):
        self.epoch=int(epoch);self.official.set_epoch(self.epoch)

    def state_dict(self):
        return dict(epoch=self.epoch,rank=self.rank,world=WORLD,official=self.official.state_dict())

    def load_state_dict(self,state):
        assert (state['rank'],state['world'])==(self.rank,WORLD)
        self.set_epoch(state['epoch']);self.official.load_state_dict(state['official'])


def next_batch(loader,sampler,iterator):
    try:return next(iterator),iterator
    except StopIteration:
        sampler.set_epoch(sampler.epoch+1)
        iterator=iter(loader)
        return next(iterator),iterator
