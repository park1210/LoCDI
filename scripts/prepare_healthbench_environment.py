"""D6 isolated environment stages. Never invokes a sampler or changes LoCDI Python."""
import argparse,email,hashlib,json,subprocess,sys,tarfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'outputs/private/healthbench_environment/D6_v1'
ENV=ROOT/'.venv-healthbench'
PINS=['blobfile==3.0.0','numpy==1.26.4','pandas==2.2.3','openai==1.109.1','jinja2==3.1.6','requests==2.32.5','tqdm==4.67.1']


def write(name,value):
    with (PRIVATE/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def read(name):return json.loads((PRIVATE/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def download_runtime():
    plan=read('runtime_source_plan.json');path=PRIVATE/plan['name']
    data=urllib.request.urlopen(plan['download_url'],timeout=60).read()
    if 'sha256:'+hashlib.sha256(data).hexdigest()!=plan['digest']:raise ValueError('RUNTIME_HASH_MISMATCH')
    with path.open('xb') as f:f.write(data)
    target=PRIVATE/'runtime';target.mkdir(exist_ok=False)
    with tarfile.open(path,'r:gz') as archive:
        for member in archive.getmembers():
            if not (target/member.name).resolve().is_relative_to(target.resolve()) or member.issym() or member.islnk():raise ValueError('UNSAFE_RUNTIME_ARCHIVE')
        archive.extractall(target)
    licenses=[{'file':p.relative_to(target).as_posix(),'sha256':sha(p)} for p in target.rglob('*') if p.is_file() and ('license' in p.name.lower() or 'copying' in p.name.lower())]
    result={'runtime_sha256':sha(path),'license_files':licenses,'executable':str(target/'python/python.exe'),'runtime_executed':False}
    write('runtime_download_receipt.json',result);return result


def resolve_wheels():
    python=PRIVATE/'runtime/python/python.exe';wheels=PRIVATE/'wheels';wheels.mkdir(exist_ok=False)
    direct=PRIVATE/'requirements.direct.txt'
    with direct.open('x',encoding='utf-8') as f:f.write('\n'.join(PINS)+'\n')
    constraints=PRIVATE/'constraints.locdi-baseline.txt'
    with constraints.open('x',encoding='utf-8') as f:f.write((ROOT/'requirements.txt').read_text(encoding='utf-8'))
    subprocess.run([str(python),'-m','pip','download','--only-binary=:all:','--dest',str(wheels),'-r',str(direct),'-c',str(constraints)],check=True)
    packages=[]
    for path in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(path) as wheel:
            metadata_name=next(n for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
            metadata=email.message_from_bytes(wheel.read(metadata_name))
            packages.append({'name':metadata['Name'],'version':metadata['Version'],'wheel':path.name,'sha256':sha(path),
                'requires_python':metadata['Requires-Python'],'requires_dist':metadata.get_all('Requires-Dist',[]),
                'license_expression':metadata['License-Expression'],'license':metadata['License'],
                'license_classifiers':[c for c in metadata.get_all('Classifier',[]) if c.startswith('License ::')],
                'license_files':[n for n in wheel.namelist() if 'license' in n.lower() or 'copying' in n.lower()]})
    write('package_license_dependency_audit.json',{'packages':packages,'installed':False,'selection':'EXACT_DIRECT_PINS_WITH_EXISTING_SDK_CONSTRAINTS_THEN_HASH_LOCK_BEFORE_INSTALL'})
    with (PRIVATE/'requirements.lock.txt').open('x',encoding='utf-8') as f:
        for p in packages:f.write(p['name']+'=='+p['version']+' --hash=sha256:'+p['sha256']+'\n')
    return {'downloaded_wheels':len(packages),'packages':[{'name':p['name'],'version':p['version'],'license_expression':p['license_expression'],'license_classifiers':p['license_classifiers']} for p in packages]}


def install():
    if ENV.exists():raise FileExistsError('ENVIRONMENT_ALREADY_EXISTS')
    audit=read('package_license_dependency_audit.json')
    if any(not any(p[k] for k in ('license_expression','license','license_classifiers','license_files')) for p in audit['packages']):raise ValueError('UNREVIEWED_LICENSE')
    for p in audit['packages']:
        if sha(PRIVATE/'wheels'/p['wheel'])!=p['sha256']:raise ValueError('WHEEL_HASH_CHANGED')
    subprocess.run([str(PRIVATE/'runtime/python/python.exe'),'-m','venv',str(ENV)],check=True)
    python=ENV/'Scripts/python.exe'
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(PRIVATE/'wheels'),'--require-hashes','-r',str(PRIVATE/'requirements.lock.txt')],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    result=json.loads(subprocess.check_output([str(python),'-c','import sys,json,importlib.metadata;print(json.dumps({"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"packages":{p.metadata["Name"]:p.version for p in importlib.metadata.distributions()}}))'],text=True))
    write('installed_environment.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('download-runtime','resolve-wheels','install'))
    args=parser.parse_args();result={'download-runtime':download_runtime,'resolve-wheels':resolve_wheels,'install':install}[args.stage]()
    print(json.dumps(result,ensure_ascii=True))
