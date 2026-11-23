"""Export existing measured results only; no policy/model execution or fitting.

Separate cohorts stay separate rows. All input paths/hashes accompany results.
Use a fresh --output directory; completed exports are never overwritten.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha


def export(output):
    output=Path(output).resolve();allowed=(ROOT/'outputs/research_evaluation').resolve()
    if allowed not in output.parents or output.is_symlink():raise ValueError('OUTPUT_NAMESPACE_REQUIRED')
    if output.exists():raise ValueError('NO_OVERWRITE')
    inputs={};rows=[]
    def source(name):
        p=ROOT/'outputs/research_evaluation'/name;inputs[name]=sha(p);return read(p)
    def add(group,condition,metric,value,patients,visits,denominator,source_name,note):
        rows.append(dict(group=group,condition=condition,metric=metric,value=value,
            patients=patients,visits=visits,denominator=denominator,
            source=source_name,scope=note,clinical_metrics='NOT_EVALUATED'))
    name='c4_22_v1/same_budget_comparison.json';m=source(name)
    for key in ['baseline','concept_diverse']:
        v=m[key]
        assert v['registered_queries']==50 and v['returned_records']==250
        for metric in ['record_precision_micro','concept_coverage_micro','record_recall_micro']:
            a=v[metric];assert abs(a['numerator']/a['denominator']-a['value'])<1e-12
            add('Memory_Original50',key,metric,a['value'],50,50,a['denominator'],name,
                'Source identity/concept reference; recall and concept coverage eligible40; not clinical relevance')
    name='c4_30_v2/diagnosis_probability_metrics.json';m=source(name)['models']
    for key in ['M0','M1','M5']:
        v=m[key]
        for metric in ['nll','brier','top1']:
            add('Probability_patient_grouped_CV',key,metric,v['visit_macro'][metric],v['patients'],v['visits'],v['visits'],name,
                'Stored OOF development results; not a new fit or independent clinical test; final M5 frozen later')
    for namespace,file in [('c4_31_v1','primary_comparison.json'),('c4_35_v1','fresh_cohort_primary_comparison.json')]:
        name=namespace+'/'+file;v=source(name);a=v['patient_macro']['top1']
        add(namespace,'H3_minus_Global_EIG','patient_macro_AUBC_difference',a['difference'],
            v['patients'],v['registered_visits'],v['patients'],name,
            'Patient cluster CI95='+json.dumps(a['ci95'])+'; Source agreement; budget1..5; separate cohort')
    name='c4_35_v1/fresh_cohort_native_policy_metrics.json';m=source(name)['metrics']
    for policy in ['voi_h2','voi_h3','global_eig','fixed']:
        v=m[policy]
        aubc=sum(v[str(b)]['patient_macro']['top1'] for b in range(1,6))/5
        add('c4_35_v1',policy,'patient_macro_Top1_AUBC',aubc,35,71,35,name,
            'Budgets1..5 averaged; fresh synthetic admitted panel, limited independence review')
        add('c4_35_v1',policy,'patient_macro_Top1_budget5',v['5']['patient_macro']['top1'],35,71,35,name,
            'Budget5 only; do not replace final diagnosis accuracy by AUBC')
    name='c4_38_v1/actual_results/independent_reaggregated_metrics.json';m=source(name)['conditions']
    for c,v in m.items():
        assert v['completed']==1 and v['generation_count']==2
        for metric in ['supported_answer_agreement','final_department_agreement']:
            a=v[metric];add('ER_Case030_development',c,metric,a['value'],1,1,a['denominator'],name,
                'Input-size selected one case; Source string agreement only; not full50/clinical accuracy')
        add('ER_Case030_development',c,'mean_executed_actions',v['mean_executed_actions_registered'],1,1,1,name,
            'Actual Ask/Test/Consult/Diagnose events; not a VoI comparison')
    name='c4_38_v1/routing_utility_eligibility.json';v=source(name)
    assert v['status']=='ROUTING_UTILITY_BLOCKED' and v['existing_fit_eligible_groups']==0
    add('Routing','existing_current_binding','eligible_complete_train_groups',0,'','','',name,
        'BLOCKED; unresolved not imputed; no fitting or additional collection')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'final_research_results.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(output/'final_research_results.json',{'rows':rows,'not_one_matched_ablation':True,
        'generation':0,'fitting':0,'Source_policy_runs':0})
    write(output/'aggregation_manifest.json',{'script_sha256':sha(Path(__file__)),
        'input_hashes':inputs,'rows':len(rows),'mode':'EXISTING_RESULTS_EXPORT_ONLY',
        'Source_probe':'not a new measurement here; original failure preserved; fixed run awaits separate approval',
        'clinical_gold_generated':0,'result_sha256':sha(output/'final_research_results.json')})
    print(json.dumps({'status':'EXPORTED_EXISTING_RESULTS','rows':len(rows),'generation':0,'fitting':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    export(p.parse_args().output)
