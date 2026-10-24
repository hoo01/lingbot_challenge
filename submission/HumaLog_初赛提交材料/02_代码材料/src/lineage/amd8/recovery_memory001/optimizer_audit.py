"""Read-only parameter sentinels using optimizer hooks; no update formula changes."""
import math


def audited_factory(factory, sentinels, local, sink, start_step=2000, observed_steps=3):
    def create(model):
        optimizer=factory(model)
        state={'calls':0,'before':{}}
        def before_step(optimizer,args,kwargs):
            if state['calls']<observed_steps:
                state['before']=sentinels(model)
        def after_step(optimizer,args,kwargs):
            state['calls']+=1
            if state['calls']<=observed_steps:
                changes={name:float((local(p.detach()).reshape(-1)[indices]-old).abs().max())
                         for name,(p,indices,old) in state['before'].items()}
                assert changes and all(math.isfinite(v) for v in changes.values())
                sink(start_step+state['calls'],changes)
                state['before']={}
        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)
        return optimizer
    return create
