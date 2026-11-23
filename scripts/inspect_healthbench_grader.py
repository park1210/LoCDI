"""Static/read-only audit; never imports or executes downloaded evaluator."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.external_dialogue import verify_source,ROOT


def inspect():
    manifest,_=verify_source();path=ROOT/manifest['files']['healthbench_eval.py']['path']
    tree=ast.parse(path.read_text(encoding='utf-8'))
    unions=sorted(n.lineno for n in ast.walk(tree) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr))
    strict=sorted(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='zip' and any(k.arg=='strict' for k in n.keywords))
    loops=sorted(n.lineno for n in ast.walk(tree) if isinstance(n,ast.While) and isinstance(n.test,ast.Constant) and n.test.value is True)
    probes={}
    for name,code in [('pep604','int | None'),('zip_strict','list(zip([], [], strict=True))')]:
        try:eval(compile(code,'<D5_builtin_compatibility_probe>','eval'),{'__builtins__':{'int':int,'zip':zip,'list':list}});probes[name]='SUPPORTED'
        except TypeError:probes[name]='UNSUPPORTED_TYPE_ERROR'
    return {'schema_version':'locdi.D5.grader_environment.v1','python':sys.version,'evaluator_imported':False,
        'evaluator_revision':manifest['files']['healthbench_eval.py']['revision'],'evaluator_sha256':manifest['files']['healthbench_eval.py']['sha256'],
        'pep604_lines':unions,'zip_strict_lines':strict,'unbounded_retry_lines':loops,'builtin_probes':probes,
        'direct_dependency_presence':{name:importlib.util.find_spec(name) is not None for name in ('blobfile','numpy','pandas')},
        'missing_downloaded_sibling_modules':[name for name in ('common.py','types.py','sampler/chat_completion_sampler.py') if not (path.parent/name).exists()],
        'separate_environment_spec':{'minimum_python_from_inspected_features':'3.10','candidate_python':'3.11',
            'full_evaluator_checkout_revision':manifest['files']['healthbench_eval.py']['revision'],
            'required_packages_observed':['blobfile','numpy','pandas','openai','transitive dependencies of pinned common/types/sampler'],
            'dependency_versions':'NOT_LOCKED_FULL_CHECKOUT_REQUIRED','environment_created':False,'compatibility_tested':False},
        'official_unmodified_execution':'BLOCKED_PYTHON_DEPENDENCIES_AND_UNBOUNDED_RETRY',
        'bounded_executor_status':'NOT_IMPLEMENTED','adapted_grading_executed':False,'grader_calls':0,'score':None}


if __name__=='__main__':print(json.dumps(inspect(),ensure_ascii=True,indent=2))
