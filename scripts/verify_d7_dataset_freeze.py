"""Read-only D7 verifier. Does not repair, rewrite, call models or install."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.dataset_freeze import verify,FreezeError


def main():
    try:result=verify();code=0
    except FreezeError as exc:result={'status':'BLOCKED','error_codes':[str(exc)],'actual_generation_calls':0,'actual_grader_calls':0};code=2
    except Exception:result={'status':'BLOCKED','error_codes':['FREEZE_VALIDATION_ERROR'],'actual_generation_calls':0,'actual_grader_calls':0};code=2
    print(json.dumps(result,ensure_ascii=True,allow_nan=False));return code


if __name__=='__main__':sys.exit(main())
