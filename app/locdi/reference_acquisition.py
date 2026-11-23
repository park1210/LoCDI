"""Independent private acquisition contract. C3-06 only exercises fake transport.

No extraction, redirects, implicit retries, model calls, or freeze integration.
Operator-pinned approval receipts are acquisition permissions, never clinical authority.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from app.locdi.llm_setup import ROOT, digest
from app.locdi.source_replay import atomic_rows

PRIVATE_ROOT=ROOT/'outputs/private/external_reference_raw'
SHA=re.compile(r'[0-9a-f]{64}')
SOURCE_HOSTS={'github.com','raw.githubusercontent.com','huggingface.co','drive.google.com','drive.usercontent.google.com'}


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('DUPLICATE_JSON_KEY')
            result[key]=value
        return result
    def constant(value):raise ValueError('NONFINITE_JSON_VALUE')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def fingerprint(plan):
    return digest({k:v for k,v in plan.items() if k!='plan_fingerprint'})


def validate_plan(plan):
    errors=[]
    if plan.get('schema_version')!='locdi.reference_acquisition_plan.v1':errors.append('PLAN_SCHEMA')
    if plan.get('plan_fingerprint')!=fingerprint(plan):errors.append('PLAN_FINGERPRINT')
    if plan.get('dataset_id') not in ('MedQA','MedMCQA','MedCaseReasoning'):errors.append('DATASET_NOT_ALLOWED')
    revision=plan.get('pinned_revision')
    if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40,64}',revision):errors.append('PINNED_DATA_REVISION_REQUIRED')
    url=plan.get('canonical_url','');parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.hostname not in SOURCE_HOSTS or parsed.port not in (None,443) or parsed.username or parsed.password or parsed.fragment:errors.append('CANONICAL_HTTPS_REQUIRED')
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',plan.get('destination_private_namespace','')):errors.append('PRIVATE_NAMESPACE_REQUIRED')
    allowed_splits={'MedQA':('train','dev','validation','test'),'MedMCQA':('train','validation','test'),'MedCaseReasoning':('train','val','test')}
    if plan.get('expected_split') not in allowed_splits.get(plan.get('dataset_id'),()):errors.append('SPLIT_REQUIRED')
    if plan.get('dataset_id')=='MedMCQA' and plan.get('expected_split')=='test':errors.append('OFFICIAL_TEST_LABELS_WITHHELD')
    for key,maximum in [('maximum_bytes',2_000_000_000),('network_request_budget',20),('timeout_seconds',120),('download_attempt_limit',1)]:
        value=plan.get(key)
        if type(value) is not int or not 1<=value<=maximum:errors.append(key.upper()+'_REQUIRED_OR_OUT_OF_RANGE')
    if not plan.get('approval_reference'):errors.append('ACQUISITION_APPROVAL_REQUIRED')
    license_=plan.get('accepted_license_use_policy_evidence')
    if not isinstance(license_,dict) or not license_.get('reference') or not SHA.fullmatch(str(license_.get('sha256',''))):errors.append('LICENSE_REVIEW_REQUIRED')
    files=plan.get('expected_files')
    if not isinstance(files,list) or not files:errors.append('EXPECTED_FILES_AND_SIZES_REQUIRED');files=[]
    names=set();total=0
    for entry in files:
        if not isinstance(entry,dict):errors.append('FILE_CONTRACT');continue
        name=entry.get('name','')
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',name) or name in ('.','..') or name in names or name=='receipt.json':errors.append('UNSAFE_OR_DUPLICATE_FILE')
        names.add(name);u=urlsplit(entry.get('url',''))
        if (u.scheme!='https' or u.hostname!=parsed.hostname or u.port not in (None,443) or u.username or u.password or u.fragment or
            not isinstance(revision,str) or (revision not in u.path and not SHA.fullmatch(str(entry.get('sha256',''))))):errors.append('UNPINNED_OR_FOREIGN_FILE_URL')
        if not SHA.fullmatch(str(entry.get('sha256',''))):errors.append('EXPECTED_SHA256_REQUIRED')
        size=entry.get('expected_bytes')
        if type(size) is not int or size<1:errors.append('EXPECTED_FILE_SIZE_REQUIRED')
        else:total+=size
    if type(plan.get('maximum_bytes')) is int and total>plan['maximum_bytes']:errors.append('MAX_BYTES_EXCEEDED')
    if type(plan.get('network_request_budget')) is int and len(files)>plan['network_request_budget']:errors.append('REQUEST_BUDGET_EXCEEDED')
    return {'status':'ACQUISITION_BLOCKED' if errors else 'PLAN_CONTRACT_VALID_NOT_APPROVED','blocker_codes':sorted(set(errors)),
            'expected_total_bytes':total,'actual_downloads':0}


class PinnedApproval:
    """Out-of-band SHA pins an operator's private approval file; no bool-status trust."""
    def __init__(self,path,expected_sha256):
        self.path=Path(path);self.expected_sha256=expected_sha256

    def verify(self,plan,now=None):
        now=now or datetime.now(timezone.utc)
        path=self.path.resolve()
        if not path.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('APPROVAL_PRIVATE_PATH')
        raw=path.read_bytes()
        if not SHA.fullmatch(str(self.expected_sha256)) or hashlib.sha256(raw).hexdigest()!=self.expected_sha256:raise ValueError('APPROVAL_HASH')
        value=strict_json(raw.decode('utf8'))
        if (value.get('schema_version')!='locdi.acquisition_approval.v1' or value.get('plan_fingerprint')!=fingerprint(plan) or
            value.get('approval_reference')!=plan.get('approval_reference') or
            value.get('license_evidence')!=plan.get('accepted_license_use_policy_evidence') or value.get('scope')!='PRIVATE_RAW_ACQUISITION_ONLY' or
            not value.get('approved_by')):raise ValueError('APPROVAL_SCOPE')
        issued=datetime.fromisoformat(value['issued_at']);expires=datetime.fromisoformat(value['expires_at'])
        if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires:raise ValueError('APPROVAL_EXPIRED_OR_NOT_YET_VALID')
        evidence=value['license_evidence'];evidence_path=(ROOT/evidence['reference']).resolve()
        if (not evidence_path.is_relative_to((ROOT/'outputs/private').resolve()) or
            hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=evidence['sha256']):raise ValueError('LICENSE_EVIDENCE_HASH')
        return {'approval_sha256':self.expected_sha256,'clinical_approval':False}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError('REDIRECT_REQUIRES_NEW_PINNED_PLAN')


def https_transport(url,timeout):
    # One request per file. HTTP errors, redirects and transient errors are terminal.
    response=build_opener(NoRedirect()).open(Request(url,headers={'Accept-Encoding':'identity'}),timeout=timeout)
    if response.geturl()!=url or response.status!=200:
        response.close();raise ValueError('UNEXPECTED_HTTP_RESPONSE')
    return response


def acquire(plan,*,execute=False,allow_download=False,authority=None,transport=None):
    """Default never dispatches. Successful private namespace requires complete receipts.

    Future operator use must supply exact file bytes/hash, accepted evidence and pinned
    approval. C3-06 CLI never enables this branch; tests use symbolic byte streams.
    """
    plan=deepcopy(plan);check=validate_plan(plan)
    if not execute:return {'mode':'DRY_RUN','plan_fingerprint':fingerprint(plan),**check}
    if check['blocker_codes']:raise ValueError('ACQUISITION_PLAN_BLOCKED')
    if allow_download is not True or not isinstance(authority,PinnedApproval):raise ValueError('EXPLICIT_DOWNLOAD_APPROVAL_REQUIRED')
    verified=authority.verify(plan)
    base=PRIVATE_ROOT.resolve()
    if not base.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('PRIVATE_DESTINATION_ESCAPE')
    # Canonical per-dataset/revision location prevents a new run name duplicating a version.
    version=base/plan['dataset_id']/plan['pinned_revision']
    if not version.resolve().is_relative_to(base):raise ValueError('PRIVATE_DESTINATION_ESCAPE')
    if version.exists():raise FileExistsError('DATASET_VERSION_ALREADY_EXISTS')
    version.mkdir(parents=True,exist_ok=False)
    receipt={'schema_version':'locdi.private.acquisition_receipt.v1','status':'STARTED',
      'dataset_id':plan['dataset_id'],'revision':plan['pinned_revision'],'split':plan['expected_split'],
      'namespace':plan['destination_private_namespace'],'plan_fingerprint':fingerprint(plan),
      'license_evidence':plan['accepted_license_use_policy_evidence'],'approval':verified,
      'files':[],'request_count':0,'attempt_limit':1,'download_complete':False}
    count=0;total=0
    try:
        for item in plan['expected_files']:
            if count>=plan['network_request_budget']:raise ValueError('NETWORK_BUDGET_EXHAUSTED')
            count+=1;receipt['request_count']=count
            response=(transport or https_transport)(item['url'],plan['timeout_seconds'])
            try:
                headers=response.headers
                if 'text/html' in headers.get('Content-Type','').lower():raise ValueError('HTML_NOT_RAW')
                length=headers.get('Content-Length')
                if length is not None and int(length)!=item['expected_bytes']:raise ValueError('CONTENT_LENGTH_MISMATCH')
                h=hashlib.sha256();size=0
                with tempfile.TemporaryFile(dir=str(version)) as temp:
                    while True:
                        # At most one byte over the tighter expected limit, not an unbounded read.
                        chunk=response.read(min(65536,item['expected_bytes']-size+1))
                        if not chunk:break
                        size+=len(chunk);total+=len(chunk)
                        if size>item['expected_bytes'] or total>plan['maximum_bytes']:raise ValueError('MAX_BYTES_EXCEEDED')
                        h.update(chunk);temp.write(chunk)
                    if size!=item['expected_bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('FILE_BYTES_OR_HASH_MISMATCH')
                    temp.seek(0)
                    with (version/item['name']).open('xb') as out:
                        while True:
                            block=temp.read(65536)
                            if not block:break
                            out.write(block)
            finally:response.close()
            receipt['files'].append({'name':item['name'],'bytes':size,'sha256':h.hexdigest(),'url':item['url'],
                'received_at':datetime.now(timezone.utc).isoformat()})
        receipt.update(status='RAW_ACQUIRED',download_complete=True)
    except Exception as exc:
        receipt.update(status='FAILED',error_type=type(exc).__name__,error_code='ACQUISITION_TERMINATED_NO_RETRY')
        atomic_rows(version,'receipt.json',receipt,jsonl=False)
        raise ValueError('ACQUISITION_FAILED_PRIVATE_RECEIPT_PRESERVED') from None
    atomic_rows(version,'receipt.json',receipt,jsonl=False)
    return receipt
