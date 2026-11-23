"""C4-34 preflight and explicitly approved single generation. No fitting/retry."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,sha,write,check
from scripts.evaluate_dynamic_horizon_source_policy import merge_pins
from app.locdi.llm_setup import digest
from app.locdi.new_cohort_execution import approval,resource_reason,tree_size,GIB

PUB=ROOT/'outputs/research_evaluation/c4_34_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_34_v1'
OLDPUB=ROOT/'outputs/research_evaluation/c4_33_v1'
OLD=ROOT/'outputs/private/research_evaluation/c4_33_v1'
RAW=PR/'raw'
CODE=['app/locdi/new_cohort_execution.py','scripts/run_new_voi_cohort.py',
      'scripts/evaluate_new_voi_cohort.py','tests/locdi/test_new_cohort_execution.py']
NAME='locdi-c4-34-generation'


def call(args,timeout=30):
    r=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=timeout)
    return {'argv':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}


def stamp():return datetime.now(timezone.utc).isoformat()


def manifest_verify(path):
    m=read(path)
    if m['fingerprint']!=digest(m['files']) or check(m['files'])['changed_or_missing']:
        raise ValueError('C433_COMPLETED_MANIFEST_CHANGED')
    return m


def generator_shell(g):
    args=['java','-Xmx3g','-jar','build/libs/synthea-with-dependencies.jar',
          '-p','1000','-o','false','-s','2026111601','-cs','2026111601','-r','20261116','-E','20261116',
          '--generate.thread_pool_size=2','--generate.max_attempts_to_keep_patient=1',
          '--generate.only_alive_patients=false','--generate.only_dead_patients=false',
          '--exporter.baseDirectory=/new','--exporter.years_of_history=0',
          '--exporter.csv.export=true','--exporter.csv.append_mode=false','--exporter.csv.folder_per_run=false',
          '--exporter.csv.included_files=','--exporter.csv.excluded_files=patient_expenses.csv',
          '--exporter.fhir.export=false','--exporter.fhir_stu3.export=false','--exporter.fhir_dstu2.export=false',
          '--exporter.ccda.export=false','--exporter.text.export=false','--exporter.json.export=true',
          '--exporter.json.include_module_history=false','--exporter.symptoms.csv.export=true',
          '--exporter.symptoms.text.export=true','--exporter.symptoms.csv.append_mode=false',
          '--exporter.symptoms.csv.folder_per_run=false','--exporter.symptoms.mode=1','Massachusetts']
    # Every shell argument is a fixed project constant; no user paths or strings.
    return ('cd /app\nsha256sum build/libs/synthea-with-dependencies.jar\n'
        'test $(sha256sum build/libs/synthea-with-dependencies.jar | cut -d\\  -f1) = '+g['generator_jar_sha256']+'\n'
        'test $(git rev-parse HEAD) = '+g['generator_commit']+'\n'
        'timeout 3600 '+' '.join(args)+'\n')


def generation_argv(g):
    return ['docker','run','--name',NAME,'--pull','never','--label','locdi.namespace=c4_34_v1',
        '--network','none','--cpus','4','--memory','6g','--memory-swap','6g','--read-only',
        '--tmpfs','/tmp:rw,size=1g','--mount','type=bind,source='+str(RAW)+',target=/new',
        '--entrypoint','/bin/sh',g['generator_image_id'],'-ec',generator_shell(g)]


def preflight():
    if PUB.exists() or PR.exists():raise ValueError('NO_PREFLIGHT_OVERWRITE')
    cm=manifest_verify(OLDPUB/'completion_manifest.json');im=manifest_verify(OLD/'private_index.json')
    pins=read(OLD/'protected_hashes.json')
    for f,m in [(OLDPUB/'completion_manifest.json',cm),(OLD/'private_index.json',im)]:
        merge_pins(pins,m['files']);merge_pins(pins,{f.relative_to(ROOT).as_posix():sha(f)})
    g=read(OLDPUB/'new_cohort_generation_protocol.json')
    if g['fingerprint']!=digest({k:v for k,v in g.items() if k!='fingerprint'}):raise ValueError('PARENT_PROTOCOL')
    PUB.mkdir(parents=True);PR.mkdir(parents=True)
    write(PR/'protected_hashes.json',pins)
    info=call(['docker','info','--format','{{json .}}'])
    images=call(['docker','images','--no-trunc','--format','{{json .}}'])
    write(PR/'docker_info_raw.json',info);write(PR/'docker_images_raw.json',images)
    if info['returncode'] or images['returncode']:raise ValueError('DOCKER_CONNECTION_BLOCKED')
    engine=json.loads(info['stdout'])
    summaries=[]
    for image in [g['generator_image_id'],'sha256:26daaf75b0220221478a499a1e04db629f1cd20830ced176288cfa4ddcfa07fb']:
        ins=call(['docker','image','inspect',image])
        if ins['returncode']:raise ValueError('PINNED_IMAGE_NOT_FOUND_NO_PULL')
        shell='sha256sum /app/build/libs/synthea-with-dependencies.jar; git -C /app rev-parse HEAD; git -C /app describe --tags --always --dirty; java -version; cat /app/src/main/resources/version.txt; cat /app/src/main/resources/synthea.properties; jar tf /app/build/libs/synthea-with-dependencies.jar'
        inspect=call(['docker','run','--rm','--pull','never','--network','none','--read-only','--cpus','1',
            '--memory','512m','--entrypoint','/bin/sh',image,'-ec',shell],timeout=60)
        write(PR/('image_'+image[7:19]+'_inspect.json'),ins)
        write(PR/('image_'+image[7:19]+'_runtime.json'),inspect)
        if inspect['returncode']:raise ValueError('READ_ONLY_IMAGE_INSPECTION_FAILED')
        lines=inspect['stdout'].splitlines();jar=lines[0].split()[0];commit=lines[1]
        props=inspect['stdout']
        missing=[key for key in ['exporter.csv.export','exporter.json.export','exporter.symptoms.mode',
            'generate.thread_pool_size','exporter.years_of_history'] if key not in props]
        if missing:raise ValueError('REQUIRED_EXPORT_OPTION_NOT_FOUND')
        summaries.append({'image_id':image,'jar_sha256':jar,'commit':commit,'describe':lines[2],
            'java':inspect['stderr'],'required_options':'PRESENT_IN_SOURCE_PROPERTIES',
            'native_JSON_exporter_class': 'org/mitre/synthea/export/JSONExporter.class' in props,
            'export_options_execution':'NOT_RUN; no patient generated'})
    selected=summaries[0]
    if selected['jar_sha256']!=g['generator_jar_sha256'] or selected['commit']!=g['generator_commit']:
        raise ValueError('PINNED_GENERATOR_MISMATCH')
    write(PUB/'docker_preflight.json',{'status':'PASS','checked_at':stamp(),'server_version':engine['ServerVersion'],
        'cpus':engine['NCPU'],'memory_bytes':engine['MemTotal'],'containers_running':engine['ContainersRunning'],
        'image_ids_present':[r['image_id'] for r in summaries], 'initial_sandbox_access':'permission denied; escalated read succeeded'})
    write(PUB/'pinned_generator_verification.json',{'status':'PASS','selected':selected,'comparison':summaries[1],
        'old_cohort_jar_matches_other_existing_image':summaries[1]['jar_sha256']=='d3b3208dc6b912fea8d60abc364a5430c5c63a6d229a7aa37468a7fcc95bdead',
        'automatic_image_switch':False,'historical_JAR_mismatch':'explained by two distinct existing images; keep C433 selected image'})
    resources={'status':'ENFORCEMENT_IMPLEMENTED_PENDING_APPROVAL','cpu_limit':4,'memory_limit_bytes':6*GIB,
        'java_heap_bytes':3*GIB,'tmpfs_limit_bytes':GIB,'engine_memory_bytes':engine['MemTotal'],
        'engine_memory_below_container_ceiling':engine['MemTotal']<6*GIB,
        'memory_semantics':'6GiB is a cgroup ceiling, not reservation; Engine5.79GiB may cause OOM below it, retain failed output if that occurs',
        'wall_seconds':3600,'output_bytes':10*GIB,'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'minimum_free_bytes':20*GIB,'output_enforcement':'host watchdog polls recursive output+log sizes every0.5sec and kills only this labeled container',
        'limit_is_not_filesystem_quota':True,'overshoot':'polling and kill latency can overshoot; actual maximum recorded; no strict instantaneous quota claim',
        'timeout_enforcement':'container timeout3600 plus independent host watchdog',
        'metrics':'docker stats CPU%, memory, PIDs, blockIO plus recursive bytes and elapsed wall',
        'cleanup':'kill only matching container if limit/error; retain stopped container inspection and every partial raw file; no retry or raw deletion'}
    if resources['disk_free_bytes']<20*GIB or engine['NCPU']<4:raise ValueError('INSUFFICIENT_HOST_RESOURCES')
    write(PUB/'generation_resource_enforcement.json',resources)
    plan={'version':'locdi.c4_34.single_generation.v1','parent_protocol_fingerprint':g['fingerprint'],
        'image_id':g['generator_image_id'],'jar_sha256':g['generator_jar_sha256'],'commit':g['generator_commit'],
        'seed':2026111601,'clinician_seed':2026111601,'population':1000,'attempts':1,'retry':0,
        'reference_date':'20261116','output_namespace':'c4_34_v1','raw_path':str(RAW),
        'argv':generation_argv(g),'resources':resources,'source_protocol_unchanged':True,
        'code_hashes':{n:sha(ROOT/n) for n in CODE}}
    plan['fingerprint']=digest(plan);write(PUB/'generation_plan.json',plan)
    write(PUB/'generation_approval_status.json',{'status':'AWAITING_EXPLICIT_USER_APPROVAL','scope':'C4_34_GENERATION_ONCE',
        'plan_fingerprint':plan['fingerprint'],'operator_evidence':None})
    for name in ['generation_execution_manifest','cohort_source_eligibility','complete_panel_report',
                 'new_cohort_policy_results','patient_paired_comparison']:
        write(PUB/(name+'.json'),{'status':'NOT_RUN','reason':'generation not approved/executed','actual_new_patients':0})
    write(PUB/'evaluation_approval_status.json',{'status':'NOT_GRANTED','scope':'C4_34_FROZEN_POLICY_EVALUATION'})
    (PUB/'generation_command.json').write_text(json.dumps(plan['argv'],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'preflight':'PASS','plan_fingerprint':plan['fingerprint'],'status':'READY_AWAITING_GENERATION_APPROVAL',
        'engine_memory_warning':resources['engine_memory_below_container_ceiling']}))


def verify_plan():
    final=PUB/'generation_plan_final.json'
    p=read(final if final.exists() else PUB/'generation_plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}) or check(p['code_hashes'])['changed_or_missing']:
        raise ValueError('EXECUTION_PLAN_OR_CODE_CHANGED')
    manifest_verify(OLDPUB/'completion_manifest.json');manifest_verify(OLD/'private_index.json')
    return p


def owned_container():
    r=call(['docker','inspect',NAME])
    if r['returncode']:return None
    item=json.loads(r['stdout'])[0]
    if item['Config'].get('Labels',{}).get('locdi.namespace')!='c4_34_v1' or item['Image']!=verify_plan()['image_id']:
        raise ValueError('CONTAINER_OWNERSHIP_MISMATCH')
    return item


def freeze_final_plan():
    p=read(PUB/'generation_plan.json')
    parent_fp=p.pop('fingerprint')
    p['initial_preflight_plan_fingerprint']=parent_fp
    p['review_note']='QA finalizes native receipt exports, overlap denominator reporting and admission code hashes; generation parameters unchanged'
    p['code_hashes']={n:sha(ROOT/n) for n in CODE}
    p['fingerprint']=digest(p)
    write(PUB/'generation_plan_final.json',p)
    write(PUB/'generation_approval_status_final.json',{'status':'AWAITING_EXPLICIT_USER_APPROVAL',
        'scope':'C4_34_GENERATION_ONCE','plan_fingerprint':p['fingerprint'],'operator_evidence':None,
        'initial_preflight_plan_superseded_for_execution':True})
    print(json.dumps({'status':'READY_AWAITING_GENERATION_APPROVAL','plan_fingerprint':p['fingerprint']}))


def generate(approval_file,execute):
    p=verify_plan()
    if not execute:
        print(json.dumps({'status':'DRY_RUN','generation':0,'argv':p['argv']}));return
    record=read(Path(approval_file)) if approval_file else None
    approval(record,'C4_34_GENERATION_ONCE',p['fingerprint'])
    if (PR/'generation_started.json').exists() or RAW.exists() or owned_container():raise ValueError('NO_OVERWRITE_NO_RESUME')
    if shutil.disk_usage(ROOT).free<20*GIB:raise ValueError('FREE_DISK_TOO_LOW')
    current=call(['docker','info','--format','{{json .}}']);engine=json.loads(current['stdout'])
    if engine['ContainersRunning'] or engine['NCPU']<4:raise ValueError('OTHER_CONTAINER_OR_CPU_CHANGE')
    write(PR/'generation_approval.json',record)
    write(PR/'generation_started.json',{'at':stamp(),'plan_fingerprint':p['fingerprint'],'approval_sha256':sha(PR/'generation_approval.json')})
    RAW.mkdir()
    start=time.monotonic();reason=None;peak=0;stats_process=None
    with (PR/'generation_console.log').open('xb') as log,(PR/'resource_samples.jsonl').open('x',encoding='utf-8') as metrics:
        process=subprocess.Popen(p['argv'],stdout=log,stderr=subprocess.STDOUT)
        try:
            last_stats=-100
            while process.poll() is None:
                elapsed=time.monotonic()-start
                size=tree_size(RAW)+(PR/'generation_console.log').stat().st_size
                peak=max(peak,size);reason=resource_reason(size,elapsed)
                sample={'elapsed_seconds':elapsed,'output_bytes':size}
                if stats_process is not None and stats_process.poll() is not None:
                    out,err=stats_process.communicate()
                    sample['docker_stats']={'returncode':stats_process.returncode,'stdout':out,'stderr':err}
                    stats_process=None
                if stats_process is None and elapsed-last_stats>=5:
                    stats_process=subprocess.Popen(['docker','stats','--no-stream','--format','{{json .}}',NAME],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    last_stats=elapsed
                elif stats_process is not None and elapsed-last_stats>10:
                    stats_process.terminate();stats_process.communicate(timeout=5);stats_process=None
                metrics.write(json.dumps(sample)+'\n');metrics.flush()
                if reason:
                    item=owned_container()
                    if item:call(['docker','kill',item['Id']])
                    break
                time.sleep(.5)
            process.wait(timeout=20)
        except Exception as e:
            reason=type(e).__name__
            item=owned_container()
            if item:call(['docker','kill',item['Id']])
            process.wait(timeout=20)
        finally:
            if stats_process is not None:
                if stats_process.poll() is None:stats_process.terminate()
                stats_process.communicate(timeout=5)
            inspect=call(['docker','inspect',NAME]);write(PR/'generation_container_final.json',inspect)
    rawpins={str(f):sha(f) for f in RAW.rglob('*') if f.is_file()}
    elapsed=time.monotonic()-start
    status='GENERATION_COMPLETED' if process.returncode==0 and reason is None else 'GENERATION_FAILED_OR_PARTIAL'
    receipt={'status':status,'at':stamp(),'plan_fingerprint':p['fingerprint'],'image_id':p['image_id'],
        'jar_sha256':p['jar_sha256'],'seed':2026111601,'attempts':1,'exit_code':process.returncode,'stop_reason':reason,
        'approval_sha256':sha(PR/'generation_approval.json'),'raw_files':rawpins,
        'elapsed_seconds':elapsed,'peak_observed_output_bytes':peak,'final_output_bytes':tree_size(RAW),
        'container_final_inspection_sha256':sha(PR/'generation_container_final.json'),
        'actual_patient_count':'NOT_YET_SOURCE_PARSED','retry':0,'partial_files_preserved':True}
    write(PR/'actual_generation_receipt.json',receipt)
    write(PUB/'actual_generation_execution.json',{k:v for k,v in receipt.items() if k not in ['raw_files','actual_patient_count']})
    print(json.dumps({'status':status,'exit_code':process.returncode,'elapsed_seconds':elapsed,'generation_attempts':1}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','generate','freeze-final-plan'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval-file');args=parser.parse_args()
    if args.stage=='preflight':preflight()
    elif args.stage=='freeze-final-plan':freeze_final_plan()
    else:generate(args.approval_file,args.execute)
