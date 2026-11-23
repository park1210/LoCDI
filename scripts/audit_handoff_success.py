"""Read saved 02903 only. No client, replay, model, fitting or file writes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.handoff_audit import audit_saved_success


if __name__=='__main__':
    try:
        result=audit_saved_success();print(json.dumps(result,ensure_ascii=True));sys.exit(0 if result['status']=='PASS' else 2)
    except Exception:
        print(json.dumps({'status':'BLOCKED','error_codes':['SAVED_HANDOFF_AUDIT_INCOMPLETE_OR_INVALID'],'new_model_generations':0}));sys.exit(2)
