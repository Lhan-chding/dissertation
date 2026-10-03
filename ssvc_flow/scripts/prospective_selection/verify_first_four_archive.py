"""Stream-verify the raw analysis archive, bindings and successful row coverage."""
import argparse,csv,gzip,hashlib,json,re,tarfile
from pathlib import Path


def main(archive,out,compact=None):
    observed={};small={};names=set();manifest=None
    with gzip.open(archive,'rb') as gz:
        with tarfile.open(fileobj=gz,mode='r|') as tar:
            for member in tar:
                assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
                assert member.name not in names;names.add(member.name)
                f=tar.extractfile(member);h=hashlib.sha256();count=0;data=[]
                keep=member.name.endswith('.json') and (member.size<2_000_000 or member.name=='RAW_MANIFEST.json')
                for block in iter(lambda:f.read(1024*1024),b''):
                    h.update(block);count+=block.count(b'\n')
                    if keep:data.append(block)
                observed[member.name]={'sha256':h.hexdigest(),'bytes':member.size,'lines':count}
                if keep:
                    value=json.loads(b''.join(data))
                    if member.name=='RAW_MANIFEST.json':manifest=value
                    else:small[member.name]=value
                if len(names)%5000==0:print('VERIFIED',len(names),flush=True)
        while gz.read(1024*1024):pass  # Validate gzip CRC and length footer to EOF.
    assert manifest is not None
    assert names=={r['name'] for r in manifest['files']}|{'RAW_MANIFEST.json'}
    source={r['source']:r['name'] for r in manifest['files']}
    for row in manifest['files']:
        assert observed[row['name']]['sha256']==row['sha256'] and observed[row['name']]['bytes']==row['bytes']
    def binding(b):
        name=source[b['path']];assert observed[name]['sha256']==b['sha256'];return observed[name]
    counts={'source_training':0,'branch_training':0,'prestate_P':0,'branch_H8':0,'branch_H32':0,'historical_E':0}
    instances={k:0 for k in counts};updates=[]
    for name,c in small.items():
        if re.fullmatch(r'campaign/(sources/6100[1-4]|branches/6100[1-4]_t(32|96)/[^/]+/repeat_1)/COMPLETE.json',name):
            assert c['status']=='TRAINING_COMPLETE';kind='source_training' if '/sources/' in name else 'branch_training'
            n=0
            for segpath in c['authoritative_segments']:
                seg=small[source[segpath]];n+=binding(seg['samples'])['lines']
                for u in seg['updates']:
                    binding(u);v=small[source[u['path']]]
                    updates.append(dict(scope=kind,path=source[u['path']],step=v['step'],loss=v['loss'],grad_norm_preclip=v['grad_norm_preclip'],sampling_seconds=v['sampling_seconds'],update_seconds=v['update_seconds']))
            assert n==c['training_outputs'];counts[kind]+=n;instances[kind]+=1
        elif re.fullmatch(r'campaign/(prestate/6100[1-4]_t(32|96)/t(24|32|88|96)|branches/6100[1-4]_t(32|96)/[^/]+/repeat_1/evaluations/H(08|32)|historical_E/[^/]+)/COMPLETE.json',name):
            assert c['status']=='EVALUATION_COMPLETE'
            kind='prestate_P' if '/prestate/' in name else 'historical_E' if '/historical_E/' in name else 'branch_H8' if '/H08/' in name else 'branch_H32'
            n=sum(binding(chunk['samples'])['lines'] for chunk in c['chunks'])
            assert n==c['generated_outputs'];counts[kind]+=n;instances[kind]+=1
    assert counts==dict(source_training=12288,branch_training=90112,prestate_P=36864,branch_H8=16896,branch_H32=202752,historical_E=36864),counts
    assert instances==dict(source_training=4,branch_training=88,prestate_P=16,branch_H8=88,branch_H32=88,historical_E=16),instances
    assert len(updates)==3200
    if compact is not None:
        data=json.loads(compact.read_text())
        for b in data['branches']:
            p=b['task']['payload']
            for h in (8,32):
                name=f"campaign/branches/{p['origin_id']}/{p['recipe_id']}/repeat_1/evaluations/H{h:02d}/COMPLETE.json"
                assert small[name]['summary']==b['evaluations'][str(h)]['summary']
        for entry in data['prestates']:
            for level,packet in entry['packets'].items():
                assert small[f"campaign/prestate/{entry['origin']}/{level}.json"]==packet
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training_updates_3200.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(updates[0]));w.writeheader();w.writerows(updates)
    receipt=dict(status='PASS',all_files_sha256_verified=len(manifest['files']),gzip_read_to_eof=True,safe_unique_regular_members=True,
        compact_summary_and_packets_match=compact is not None,authoritative_rows=counts,instances=instances,total_authoritative_rows=sum(counts.values()),finite_update_rows=len(updates),
        missing_images=manifest['missing_images'],omitted_files=len(manifest['omitted_files']),
        scope='Archive hashes and committed raw-file bindings/line counts; prior semantic identity audits retained. Not a fresh re-score of raw text.')
    assert all(__import__('math').isfinite(r[k]) for r in updates for k in ['loss','grad_norm_preclip'])
    (out/'RAW_CONTENT_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('out',type=Path);p.add_argument('--compact',type=Path);a=p.parse_args();main(a.archive,a.out,a.compact)
