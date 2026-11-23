# -*- coding: utf-8 -*-
"""New, separately versioned literature facts. Existing knowledge corpus untouched.

Original summaries of primary pages read on 2026-11-22. Not clinical judgments.
No case, Source label, model response or performance result is an input here.
"""
from copy import deepcopy
from app.locdi.llm_setup import digest
VERSION='original50_literature_evidence.v1'
CHECKED='2026-11-22'
# publication_year is intentionally UNKNOWN: a review date is not a publication date.
_S=[
('R01','NHLBI/NIH','Coronary heart disease — Symptoms','https://www.nhlbi.nih.gov/health/coronary-heart-disease/symptoms','2024-12-27','Symptoms may be silent or occur with exertion.'),
('R02','NHLBI/NIH','Coronary heart disease — Diagnosis','https://www.nhlbi.nih.gov/health/coronary-heart-disease/diagnosis','2024-12-27','Clinical assessment and appropriate heart tests are combined.'),
('R03','NHLBI/NIH','Atrial fibrillation — Symptoms','https://www.nhlbi.nih.gov/health/atrial-fibrillation/symptoms','2022-11-30','Fatigue and episodic palpitations can occur; symptoms can be absent.'),
('R04','NHLBI/NIH','Atrial fibrillation — Diagnosis','https://www.nhlbi.nih.gov/health/atrial-fibrillation/diagnosis','UNKNOWN','ECG or longer rhythm monitoring supports assessment.'),
('R05','NHLBI/NIH','Metabolic syndrome — Symptoms','https://www.nhlbi.nih.gov/health/metabolic-syndrome/symptoms','2022-05-18','Several components are silent; hyperglycemia can cause thirst or fatigue.'),
('R06','NHLBI/NIH','Metabolic syndrome — Diagnosis','https://www.nhlbi.nih.gov/health/metabolic-syndrome/diagnosis','2022-05-18','Assessment uses waist, blood pressure, glucose, HDL and triglycerides.'),
('R07','NHLBI/NIH','High blood triglycerides','https://www.nhlbi.nih.gov/health/high-blood-triglycerides','2023-04-19','Often silent; assess measured triglycerides and repeat/context when needed.'),
('R08','NHLBI/NIH','Blood cholesterol — Symptoms','https://www.nhlbi.nih.gov/health/blood-cholesterol/symptoms','2024-04-18','Unhealthy cholesterol usually has no symptoms.'),
('R09','NHLBI/NIH','Blood cholesterol — Diagnosis','https://www.nhlbi.nih.gov/health/blood-cholesterol/diagnosis','2024-04-18','A lipid panel and clinical context are used.'),
('R10','NIDDK/NIH','Symptoms & causes of diabetes','https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes','2024-10','Thirst, urination, fatigue or blurred vision can occur; type2 may be unnoticed.'),
('R11','NIDDK/NIH','The A1C test & diabetes','https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test','2018-04','A1C reflects about three months; confirm abnormal diagnosis without clear symptoms.'),
('R12','NHS','Alzheimer disease — Symptoms','https://www.nhs.uk/conditions/alzheimers-disease/symptoms/','2024-07-04','Gradual memory and functioning changes; the course varies.'),
('R13','NHS','Alzheimer disease — Diagnosis','https://www.nhs.uk/conditions/alzheimers-disease/diagnosis/','2024-07-04','Cognitive/functional assessment and exclusion of other causes are needed.'),
('R14','NLM/NIH MedlinePlus Genetics','Alzheimer disease','https://medlineplus.gov/genetics/condition/alzheimers-disease/','2019-05-01','Some familial early forms involve APP, PSEN1 or PSEN2; onset age alone is insufficient.'),
('R15','NIDDK/NIH','Identify & evaluate CKD','https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/kidney-disease/identify-manage-patients/evaluate-ckd','UNKNOWN','Assess eGFR, urine albumin and chronicity exceeding three months.'),
('R16','National Kidney Foundation','Stage G2 CKD','https://www.kidney.org/kidney-topics/stage-g2-chronic-kidney-disease-ckd','2026-07-17','eGFR60-89 requires lasting kidney damage; symptoms are usually absent.'),
('R17','National Kidney Foundation','Stage G3a CKD','https://www.kidney.org/kidney-topics/stage-g3a-chronic-kidney-disease-ckd','2026-07-17','eGFR45-59 for at least three months; symptoms often absent or nonspecific.'),
('R18','National Kidney Foundation','Stage G3b CKD','https://www.kidney.org/kidney-topics/stage-g3b-chronic-kidney-disease-ckd','2026-07-17','eGFR30-44 with chronicity; fatigue, itch or urinary changes may occur.'),
('R19','National Kidney Foundation','Stage G4 CKD','https://www.kidney.org/kidney-topics/stage-g4-chronic-kidney-disease-ckd','2026-07-17','eGFR15-29 with chronicity; symptoms and complications require assessment.'),
('R20','NIAMS/NIH','Osteoporosis — Overview and symptoms','https://www.niams.nih.gov/health-topics/osteoporosis','UNKNOWN','Frequently silent before fracture; fracture symptoms do not establish every case.'),
('R21','NIAMS/NIH','Osteoporosis — Diagnosis','https://www.niams.nih.gov/health-topics/osteoporosis/diagnosis-treatment-and-steps-to-take','2022-12','DXA at hip/spine, fracture context and demographic interpretation matter.'),
('R22','NHLBI/NIH','COPD — Overview','https://www.nhlbi.nih.gov/health/copd','UNKNOWN','COPD includes bronchitis and emphysema; structural subtype evidence is separate.'),
('R23','NHLBI/NIH','COPD — Symptoms','https://www.nhlbi.nih.gov/health/copd/symptoms','UNKNOWN','Cough, sputum and exertional breathlessness may develop gradually; early disease may be mild.'),
('R24','NHLBI/NIH','COPD — Diagnosis','https://www.nhlbi.nih.gov/health/copd/diagnosis','2024-10-04','Spirometry and clinical/exposure context; appropriate imaging may help assessment.'),
('R25','NHLBI/NIH','Asthma — Symptoms','https://www.nhlbi.nih.gov/health/asthma/symptoms','2024-04-17','Variable wheeze, nocturnal cough, breathlessness or chest tightness.'),
('R26','NHLBI/NIH','Asthma — Diagnosis','https://www.nhlbi.nih.gov/health/asthma/diagnosis','2024-04-17','Spirometry, PEF and bronchodilator responsiveness; other tests depend on context.'),
]

def feature(key,text,sources):return {'feature_id':key,'text':text,'sources':sources}
def test(key,name,sources,codes=()):return {'test_id':key,'name':name,'sources':sources,'matching_source_codes':list(codes)}
F={
'exertional_chest':feature('exertional_chest','Chest discomfort during physical activity.',['R01']),
'exertional_dyspnea':feature('exertional_dyspnea','Breathlessness during activity.',['R01','R23']),
'palpitations':feature('palpitations','Intermittent fluttering or racing heartbeat.',['R03']),
'fatigue_af':feature('fatigue_af','Unusual tiredness.',['R03']),
'dizziness':feature('dizziness','Episodes of dizziness.',['R03']),
'thirst':feature('thirst','Feeling thirstier than usual.',['R10']),
'polyuria':feature('polyuria','Urinating more often than usual.',['R10']),
'fatigue_glucose':feature('fatigue_glucose','Tiredness without a clear explanation.',['R05','R10']),
'blurred_vision':feature('blurred_vision','Episodes of blurred vision.',['R10']),
'memory_lapse':feature('memory_lapse','Difficulty recalling recent conversations or events.',['R12']),
'word_finding':feature('word_finding','Difficulty finding a familiar word.',['R12']),
'daily_task':feature('daily_task','More difficulty managing familiar everyday tasks.',['R14']),
'renal_fatigue':feature('renal_fatigue','Feeling tired more readily than before.',['R17','R18','R19']),
'itch':feature('itch','Itchy or dry skin.',['R17','R18','R19']),
'urination_change':feature('urination_change','A change in the usual frequency of urination.',['R17','R18','R19']),
'ankle_swelling':feature('ankle_swelling','Swelling around the ankles.',['R19']),
'nausea':feature('nausea','Nausea or reduced appetite.',['R17','R18','R19']),
'wheeze':feature('wheeze','Episodes of wheezing.',['R25']),
'night_cough':feature('night_cough','Cough more noticeable at night.',['R25']),
'chest_tight':feature('chest_tight','Intermittent chest tightness.',['R25']),
'sputum_cough':feature('sputum_cough','An ongoing cough with sputum.',['R22','R23']),
'respiratory_fatigue':feature('respiratory_fatigue','Tiredness accompanying a breathing concern.',['R23']),
}
T={
'ecg':test('ecg','Electrocardiogram / rhythm assessment',['R02','R04']),
'ischemia':test('ischemia','Clinically indicated coronary or stress assessment',['R02']),
'a1c':test('a1c','Hemoglobin A1c',['R11'],['4548-4']),
'glucose':test('glucose','Plasma glucose with sampling context',['R06','R11'],['2345-7','2339-0']),
'bp':test('bp','Repeated blood-pressure assessment',['R06'],['8462-4','8480-6']),
'waist':test('waist','Waist measurement',['R06'],['56115-9']),
'triglycerides':test('triglycerides','Blood triglycerides',['R07'],['2571-8']),
'lipids':test('lipids','Lipid profile, including cholesterol fractions',['R09'],['2093-3','2085-9','13457-7','18262-6']),
'cognition':test('cognition','Cognitive and functional assessment',['R13']),
'dementia_workup':test('dementia_workup','Clinical and indicated blood/imaging assessment for other causes',['R13']),
'genetic_review':test('genetic_review','Family history and indicated genetic evaluation',['R14']),
'egfr':test('egfr','Creatinine/eGFR and prior renal measurements',['R15'],['38483-4','33914-3']),
'uacr':test('uacr','Urine albumin-to-creatinine ratio',['R15']),
'dxa':test('dxa','Hip/spine DXA bone-density assessment',['R21']),
'spirometry':test('spirometry','Spirometry / lung-function testing',['R24','R26']),
'bronchodilator':test('bronchodilator','Bronchodilator responsiveness or PEF when indicated',['R26']),
'lung_imaging':test('lung_imaging','Clinically indicated lung imaging',['R22','R24']),
}

def card(label,symptoms,tests,sources,early,interpretation,confirmation,time_course,asymptomatic,differential='NOT_VERIFIED_IN_SELECTED_PRIMARY_SOURCES'):
    return dict(disease=label,symptom_feature_ids=symptoms,candidate_test_ids=tests,sources=sources,
        early_or_atypical=early,interpretation=interpretation,differential=differential,
        confirmation_needed=confirmation,time_course=time_course,asymptomatic_or_uncertain=asymptomatic,
        clinical_reviewed=False,clinical_validity='UNREVIEWED')

C=[
card('Ischemic heart disease (disorder)',['exertional_chest','exertional_dyspnea'],['ecg','ischemia'],['R01','R02'],'증상이 없거나 활동 관련 불편만 있을 수 있음.','위험인자·활력징후 하나만으로 확진 불가.','증상/병력 및 적응증에 맞는 심장검사.','장기간 진행과 활동·휴식 관련 변화를 확인.','silent presentation 가능.'),
card('Atrial fibrillation (disorder)',['palpitations','fatigue_af','dizziness'],['ecg'],['R03','R04'],'간헐적이거나 무증상일 수 있음.','맥박수 하나는 리듬 진단이 아님.','실제 ECG/리듬 기록 및 원인 평가.','증상 빈도·기간 및 반복 리듬 관찰.','증상 유무만으로 배제·확진 불가.'),
card('Metabolic syndrome X (disorder)',['fatigue_glucose'],['bp','waist','glucose','lipids','triglycerides'],['R05','R06'],'여러 구성요소가 조용히 존재할 수 있음.','혈압·혈당·지질·복부비만의 조합; 단일 수치로 확정하지 않음.','반복 측정/공복·치료 상태 및 여러 구성요소.','반복 관찰이 필요; 환자별 진행 시점은 미확인.','무증상 가능.'),
card('Osteoporosis (disorder)',[],['dxa'],['R20','R21'],'골절 전에는 증상이 없을 수 있음.','DXA와 골절/연령·성별 맥락; 보편적 임계값 자동 적용 안 함.','골밀도 및 실제 골절/위험인자 검토.','일반 검사만으로 진행 판단 안 함.','골절·요통을 모든 사례에 넣지 않음.'),
card('Hypertriglyceridemia (disorder)',[],['triglycerides'],['R07'],'대개 뚜렷한 자각 증상이 없음.','성인 공복 TG 지속150mg/dL 이상 등 문맥 검토; 합성 수치 생성 안 함.','실제 중성지방 측정·공복/반복·원인 확인.','지속성 및 채혈 조건 확인.','무증상 허용.'),
card('Diabetes mellitus type 2 (disorder)',['thirst','polyuria','fatigue_glucose','blurred_vision'],['a1c','glucose'],['R10','R11'],'증상이 서서히 나타나거나 인지되지 않을 수 있음.','A1c6.5% 이상은 명확한 증상 없는 경우 확인 검사; 유형 판단은 별개.','반복 검사 및 임상/당뇨 유형 검토.','A1c는 약3개월 평균을 반영.','무증상/비특이적 호소 허용.','type1/다른 원인과 구분은 별도 검토.'),
card('Hyperlipidemia (disorder)',[],['lipids'],['R08','R09'],'자각 증상 없는 경우가 흔함.','실제 지질검사와 전체 위험 평가.','LDL/HDL 등 지질 분획 및 관련 병력.','반복 수치와 맥락; 임의 진행률 없음.','무증상 허용.'),
card("Alzheimer's disease (disorder)",['memory_lapse','word_finding','daily_task'],['cognition','dementia_workup'],['R12','R13','R14'],'초기 기억/언어 변화 또는 정보가 불충분한 호소.','단일 기억 점수·혈액값만으로 확진 불가.','기능·인지 평가 및 다른 원인 배제.','수년에 걸친 점진 변화 가능; 개인별 경과는 생성 안 함.','현재 병력 부족 상태도 허용.','우울·불안·약물 등 대체 원인은 별도 평가.'),
card("Familial Alzheimer's disease of early onset (disorder)",['memory_lapse','word_finding'],['cognition','dementia_workup','genetic_review'],['R12','R13','R14'],'인지 변화만으로 가족성·발병연령은 확정 불가.','APP/PSEN1/PSEN2 관련 가능성; 변이나 가족력을 생성하지 않음.','실제 발병 병력·가족/유전 근거.','실제 onset은 미검토; 현재 연령으로 추정하지 않음.','희소한 병력/불확실성 유지.'),
card('Chronic kidney disease stage 2 (disorder)',[],['egfr','uacr'],['R15','R16'],'대개 증상 없음.','eGFR60-89 단독은 불충분; 신장손상 및3개월 지속성 필요.','uACR/기타 손상과 종전 결과.','현재 한 시점만으로 만성성 확정 안 함.','무증상 허용.','급성 변화와 만성성 구분 필요.'),
card('Chronic kidney disease stage 3 (disorder)',['renal_fatigue','itch','urination_change'],['egfr','uacr'],['R15','R17','R18'],'초기에는 없거나 비특이적.','G3a45-59/G3b30-44와3개월 지속성; subtype 자동 배정 안 함.','반복 eGFR·소변 및 임상 평가.','반복 검사로 급성 저하와 구분.','정보 부족/무증상 허용.','급성 신손상과 구분.'),
card('Chronic kidney disease stage 4 (disorder)',['renal_fatigue','itch','ankle_swelling','nausea'],['egfr','uacr'],['R15','R19'],'피로 등 비특이적 호소; 모든 증상을 강제하지 않음.','eGFR15-29 및3개월 지속성.','반복 기능·손상 및 합병증 평가.','한 개 비정상 검사로 만성질환 확진 안 함.','병력 불충분 상태도 유지.','급성 신손상과 구분.'),
card('Chronic obstructive bronchitis (disorder)',['sputum_cough','exertional_dyspnea'],['spirometry'],['R22','R23','R24'],'초기 약한 호소 또는 증상 부재 가능.','COPD 문헌을 사용; 정확한 subtype 확증은 별도.','기류·노출·병력 평가.','서서히 진행/악화 가능; 개인 기간은 생성 안 함.','전형적 기침을 모든 사례에 강제 안 함.'),
card('Pulmonary emphysema (disorder)',['exertional_dyspnea','respiratory_fatigue'],['spirometry','lung_imaging'],['R22','R23','R24'],'초기 경미/무증상 가능.','구조적 폐포 손상 subtype; 호흡수만으로 구분 안 됨.','기류·임상·필요시 영상 등 subtype 근거.','진행성 경과 가능; 기간/CT 결과 생성 안 함.','불확실성을 유지.'),
card('Asthma (disorder)',['wheeze','night_cough','chest_tight'],['spirometry','bronchodilator'],['R25','R26'],'간헐적·야간/아침 증상 가능.','변동성 기류 제한/약물 반응은 실제 검사로 확인.','실제 폐기능·반응 및 임상 평가.','하루 내/기간에 따라 변동 가능.','현재 증상 없는 구간도 허용.'),
]

def evidence_catalog():
    sources={key:{'institution':inst,'title':title,'url':url,'publication_year':'UNKNOWN',
        'revision_or_review_date':rev,'checked_on':CHECKED,'verification':'PRIMARY_PAGE_CONTENT_OBSERVED',
        'observed_support_summary':summary,'raw_page_bytes_sha256':'NOT_CAPTURED','license_policy':'ORIGINAL_PARAPHRASE_AND_LINK_ONLY'}
        for key,inst,title,url,rev,summary in _S}
    cards=[]
    for i,c in enumerate(C,1):cards.append({'concept_id':'LOCDI_AUG_%02d'%i,'concept_system':'LOCAL_NOT_EXTERNAL_TERMINOLOGY',**c})
    data={'version':VERSION,'checked_on':CHECKED,'sources':sources,'features':F,'tests':T,'cards':cards,
        'failed_access_not_used':[
            {'url':'https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/what-are-signs-alzheimers-disease','observed_error':'405'},
            {'url':'https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed','observed_error':'405'},
            {'url':'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/symptoms-causes','observed_error':'404'}],
        'clinical_reviewed':False}
    return deepcopy(data)
