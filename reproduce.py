"""Rebuild the saved-run analysis, table and figure on CPU. No model or network."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
COMPONENTS = ['own', 'other', 'context', 'source']
N_BOOT = 10_000
BOOT_SEED = 2026110608

def load_json(p):
    return json.loads(p.read_text())

def ci(values, indices):
    means = values[indices].mean(axis=1)
    return {'mean_nats':float(values.mean()),
            'ci95_nats':np.quantile(means,[.025,.975]).tolist()}

def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    data = np.load(ROOT/'raw_scores.npz', allow_pickle=False)
    raw = data['scores']
    offsets = data['offsets']
    protocol = load_json(ROOT/'protocol.json')
    saved = load_json(ROOT/'analysis.json')
    target = saved['arms']['lr_5e5']
    assert raw.shape == (200, 2, len(offsets), 3)
    assert np.isfinite(raw).all()
    assert offsets.tolist() == protocol['probe_offsets']
    assert len(np.unique(data['pair_ids'])) == 200
    window = np.flatnonzero(np.isin(offsets, protocol['persistent_offsets']))
    baseline = int(np.flatnonzero(offsets == -1)[0])
    changes = raw - raw[:, :, baseline:baseline+1, :]
    paired = changes[:, 0] - changes[:, 1]
    effects = {'own':paired[:,:,0], 'other':paired[:,:,1],
               'context':paired[:,:,0]-paired[:,:,1], 'source':paired[:,:,2]}
    indices = np.random.default_rng(BOOT_SEED).integers(0, len(raw), (N_BOOT, len(raw)))
    summaries, curves = {}, {}
    maximum_difference = 0.0
    comparisons = 0
    def compare(a,b):
        nonlocal maximum_difference, comparisons
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        maximum_difference=max(maximum_difference,float(np.max(np.abs(a-b))))
        if not np.allclose(a,b,rtol=0,atol=1e-10):
            raise AssertionError(f'Saved-score verification failed: {a} != {b}')
        comparisons += a.size
    for component in COMPONENTS:
        matrix = effects[component]
        summaries[component] = ci(matrix[:,window].mean(axis=1), indices)
        curves[component] = [dict(offset=int(offset),**ci(matrix[:,i],indices))
                             for i,offset in enumerate(offsets)]
        for field in ['mean_nats','ci95_nats']:
            compare(summaries[component][field],target['components'][component][field])
            compare([r[field] for r in curves[component]],
                    [r[field] for r in target['curves'][component]])
        # Verify every underlying pair, not just a figure drawn from saved means.
        compare(matrix,np.asarray([r['effects'][component] for r in target['pair_effects']]))
    mask = ~data['flagged']
    si = np.random.default_rng(2026110609).integers(0,int(mask.sum()),(N_BOOT,int(mask.sum())))
    sensitivity = ci(effects['context'][mask][:,window].mean(axis=1),si)
    for k in ['mean_nats','ci95_nats']:
        compare(sensitivity[k],target['source_quality_sensitivity']['components']['context'][k])
    events = np.unique(data['events'])
    cluster = np.asarray([effects['context'][data['events']==e][:,window].mean() for e in events])
    ei = np.random.default_rng(2026110610).integers(0,len(cluster),(N_BOOT,len(cluster)))
    event_summary = ci(cluster,ei)
    for k in ['mean_nats','ci95_nats']:
        compare(event_summary[k],target['event_cluster_sensitivity'][k])
    direct = data['directional']
    correct = direct[:,:,:,:,0] > direct[:,:,:,:,1]
    accuracy_change = correct[:,:,1].astype(int)-correct[:,:,0].astype(int)
    directional = {'treated':float(accuracy_change[:,0].mean()),
                   'control':float(accuracy_change[:,1].mean())}
    directional['treated_minus_control']=directional['treated']-directional['control']
    for k,v in directional.items():compare(v,target['actual_directional_accuracy_change'][k])
    summary = {'components':summaries,'curves':curves,'pairs':len(raw),
               'persistent_offsets':offsets[window].tolist(),
               'source_quality_sensitivity':sensitivity,
               'event_cluster_sensitivity':event_summary,
               'directional_accuracy_change':directional,
               'x_star_nats':saved['x_star_nats'],
               'advance_chase':summaries['context']['ci95_nats'][0]>0}
    assert summary['advance_chase'] == target['gate']['advance_chase'] == False
    (OUT/'recomputed_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'results_table.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['measure','mean_nats','lower_95','upper_95'])
        for c in ['source','own','other','context']:
            s=summaries[c]
            writer.writerow([c,s['mean_nats'],*s['ci95_nats']])
    labels={'source':'Exposure excerpt', 'own':'Appropriate probe context',
            'other':'Mismatched probe context', 'context':'Context discrimination (own - other)'}
    lines=['| Measure | Change (nats) | 95% interval |','|---|---:|---:|']
    for c in ['source','own','other','context']:
        s=summaries[c];lo,hi=s['ci95_nats']
        lines.append(f"| {labels[c]} | {s['mean_nats']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    (OUT/'results_table.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#263447','text.color':'#172638',
                         'xtick.color':'#465466','ytick.color':'#465466',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.0),sharex=True,sharey=True)
    colors={'source':'#187D80','own':'#3068AD','other':'#C67A28','context':'#704B97'}
    for letter,ax,c in zip('ABCD',axes.flat,['source','own','other','context']):
        rows=curves[c];means=np.array([r['mean_nats'] for r in rows])
        bounds=np.array([r['ci95_nats'] for r in rows])
        ax.axvspan(10,50,color='#F2F4F7',zorder=0)
        ax.axhline(0,color='#687788',lw=.9)
        ax.axvline(0,color='#8994A1',ls=':',lw=.9)
        ax.fill_between(offsets,bounds[:,0],bounds[:,1],color=colors[c],alpha=.17,lw=0)
        ax.plot(offsets,means,color=colors[c],lw=2,marker='o',ms=3.3)
        ax.set_title(f'{letter}  {labels[c]}',loc='left',fontweight='bold',fontsize=10.5,pad=9)
        s=summaries[c];lo,hi=s['ci95_nats']
        ax.text(.03,.94,f"Mean +10 to +50: {s['mean_nats']:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                transform=ax.transAxes,fontsize=9.1,va='top',color=colors[c])
        if c=='context':
            ax.axhline(saved['x_star_nats'],color='#8994A1',ls='--',lw=1)
            ax.text(2,saved['x_star_nats']+.025,'Frozen sensitivity anchor: 0.578',fontsize=8.3,color='#687788')
        ax.set_xlim(-2.5,52);ax.set_ylim(-.20,.97)
        ax.set_xticks([0,10,20,30,40,50]);ax.set_yticks([-.2,0,.2,.4,.6,.8])
        ax.grid(axis='y',color='#DFE4EA',lw=.5,alpha=.7)
    fig.suptitle('What changed after one rare-word exposure?',x=.105,ha='left',y=.985,
                 fontweight='bold',fontsize=18)
    fig.text(.105,.927,'200 matched pairs | one completed arm, learning rate 5e-5 | one training seed',
             color='#526174',fontsize=10)
    fig.text(.02,.54,'Change from before exposure, treated minus control (nats)',
             rotation=90,va='center',fontsize=10)
    fig.supxlabel('Updates relative to each word\'s exposure (0 = just after the exposure update)',
                  y=.09,fontsize=10)
    fig.text(.105,.035,'Bands: pointwise 95% paired-bootstrap intervals, not variation across training seeds.\n'
             'Shaded region: the frozen +10 to +50 summary window. Target scores sum subtoken log probabilities.',
             fontsize=9,color='#526174',linespacing=1.5)
    fig.subplots_adjust(left=.105,right=.98,top=.86,bottom=.18,hspace=.30,wspace=.14)
    fig.savefig(OUT/'rare_word_learning_curves.png',dpi=220,facecolor='white')
    fig.savefig(OUT/'rare_word_learning_curves.pdf',facecolor='white')
    plt.close(fig)
    verification={'status':'PASS','numeric_comparisons':int(comparisons),
                  'maximum_absolute_difference':maximum_difference,
                  'elapsed_seconds':time.perf_counter()-started,
                  'python':sys.version.split()[0],'numpy':np.__version__,
                  'matplotlib':matplotlib.__version__,
                  'raw_scores_sha256':hashlib.sha256((ROOT/'raw_scores.npz').read_bytes()).hexdigest(),
                  'scope':'Numerical reconstruction from saved observations; no training, model loading, or network access.'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
