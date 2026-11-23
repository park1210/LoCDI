"""Fail before inference on unsafe Windows checkpoint layouts. No OS changes."""
from pathlib import Path

MAX_FILE_CHARS=239  # Conservative reserve below MAX_PATH260, including temp names.
RESERVED_LEAF_CHARS=64
FILES=('Request0012_before.json','Request0012_after.json','Request0012_core_response.json',
       'runtime_snapshot_private.json','raw_runner_result.json','actual_core_decisions.json','episode_and_policy.json')

def check_directory(directory):
    directory=Path(directory).resolve()
    longest=max([len(str(directory/name)) for name in FILES]+[len(str(directory))+1+RESERVED_LEAF_CHARS])
    if longest>MAX_FILE_CHARS:raise ValueError('STORAGE_PATH_BUDGET_EXCEEDED_BEFORE_GENERATION')
    return {'directory_chars':len(str(directory)),'reserved_max_file_chars':longest,
            'limit':MAX_FILE_CHARS,'known_longest_file_chars':max(len(str(directory/name)) for name in FILES)}

def validation_layout(root,agents,patients=2):
    return [check_directory(Path(root)/('Patient%02d'%i)/a['department']/'Episode0001')
            for i in range(1,patients+1) for a in agents]

def evaluation_layout(root,episodes):
    return check_directory(Path(root)/'requests'/('Episode%04d'%episodes))
