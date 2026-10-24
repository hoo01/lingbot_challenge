"""Explicit source-only frame and original-seen-instruction selection."""
import hashlib,json
import numpy as np

def pool_sha(pool):
    return hashlib.sha256(json.dumps(pool,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

class FilterPlan:
    def __init__(self,records,path):
        self.plan=json.loads(path.read_text());self.sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert self.plan['original_frames']==549787 and self.plan['filtered_frames']==549737
        starts=[];offset=0;self.kept={};removed=[]
        rows={(r['task'],int(r['episode'])):i for i,r in enumerate(records)}
        assert len(records)==len(rows)==2500
        for row in self.plan['instruction_filters']:
            key=(row['task'],int(row['episode']));record=records[rows[key]];pool=record['instructions']
            assert len(pool)==row['original_pool_entries'] and pool_sha(pool)==row['original_pool_sha256'],key
            keep=row['keep_original_indices'];exclude=row['exclude_conflicting_indices']
            assert sorted(keep+exclude)==list(range(len(pool))) and len(set(keep))==len(keep) and keep
            assert row['new_instruction_texts_generated']==row['other_episode_instructions_used']==0
            self.kept[key]=list(keep)
        for record in records:
            n=int(record['frames']);ids=np.arange(offset,offset+n,dtype=np.int64)
            if record['task']=='place_shoe' and int(record['episode'])==45:
                assert record['sha256']=='c3697c3bba6962487f9f04ba2bb284a028d62360ce12379504daae8d2465974c'
                mask=(np.arange(n)<95)|(np.arange(n)>144);removed.extend(ids[~mask].tolist());ids=ids[mask]
            starts.extend(ids.tolist());offset+=n
        self.original_indices=np.asarray(starts,dtype=np.int64)
        assert offset==549787 and len(starts)==549737 and len(removed)==50
        assert np.all(np.diff(self.original_indices)>0)
        self.removed=removed

    def instruction_indices(self,record):
        return self.kept.get((record['task'],int(record['episode'])),list(range(len(record['instructions']))))
