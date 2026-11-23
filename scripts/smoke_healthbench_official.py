"""Python 3.11 only: unmodified official imports/pure math, TEST_ONLY inputs."""
import hashlib,importlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/private/healthbench_environment/D6_v1/vendor/simple-evals-652c89d0ca9df547706735883097e9537d40dc47'


def main():
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('D6_NO_NETWORK_OR_MODEL_CALLS')
    sys.addaudithook(guard)
    expected='b763d8c1f53ecd165f0d170a32bc948216b3dfeadafe238b2083328e1af4b2df'
    assert hashlib.sha256((SOURCE/'healthbench_eval.py').read_bytes()).hexdigest()==expected
    package=types.ModuleType('d6_official');package.__path__=[str(SOURCE)];sys.modules['d6_official']=package
    evaluator=importlib.import_module('d6_official.healthbench_eval')
    tests=importlib.import_module('d6_official.healthbench_eval_test')
    tests.test_calculate_score();tests.test_calculate_length_adjusted_score()
    fixtures=[]
    for points,flags in [([7,5,10,-6],[True,False,True,True]),([1,-4],[True,True]),([1,-4],[False,False]),([-1],[True]),([1,2],[True,False])]:
        rubrics=[evaluator.RubricItem('TEST_ONLY',p,[]) for p in points]
        value=evaluator.calculate_score(rubrics,[{'criteria_met':b} for b in flags])
        fixtures.append({'points':points,'criteria_met':flags,'test_only_score':value})
    try:evaluator.calculate_score([evaluator.RubricItem('TEST_ONLY',1,[])],[])
    except ValueError:mismatch='VALUE_ERROR'
    else:raise AssertionError('Official strict zip should reject a missing grade')
    np=evaluator.np;np.random.seed(42)
    first=float(evaluator._compute_clipped_stats([-2.,2.],'bootstrap_std'));np.random.seed(42)
    second=float(evaluator._compute_clipped_stats([-2.,2.],'bootstrap_std'));assert first==second
    assert evaluator._compute_clipped_stats([-2.,2.],'mean')==0
    aggregate=evaluator._aggregate_get_clipped_mean([
        evaluator.SingleEvalResult(score=-2.,metrics={'TEST_ONLY_metric':-2.}),
        evaluator.SingleEvalResult(score=2.,metrics={'TEST_ONLY_metric':2.})])
    assert aggregate.score==0 and aggregate.metrics['score:n_samples']==2
    result={'schema_version':'locdi.D6.official_environment_smoke.v1','python':sys.version,'interpreter':sys.executable,
        'official_module_import':'PASS','source_sha256':expected,'official_pure_tests':'2_PASS',
        'test_only_arithmetic_fixtures':fixtures,'strict_length_mismatch':mismatch,
        'dataset_mean_clipped_after_averaging':True,'test_only_bootstrap_std':first,'bootstrap_draws':1000,
        'bootstrap_repeat_with_fixed_numpy_seed_matches':True,'grade_sample_called':False,'sampler_constructed':False,
        'network_disabled':True,'actual_generation_calls':0,'actual_grader_calls':0,'official_healthbench_score':None}
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
