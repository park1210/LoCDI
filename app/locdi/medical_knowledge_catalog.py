"""Patient-independent original summaries of checked public primary sources.

No Source labels, evaluation cases or clinical review judgments are inputs.
Local concept IDs are explicit; external terminology mapping is not invented.
"""
from copy import deepcopy

VERSION='medical_catalog_public_summary.v1'
DATE='2026-11-21'
SOURCES={
 'S01':('NHLBI/NIH','Coronary heart disease diagnosis','https://www.nhlbi.nih.gov/health/coronary-heart-disease/diagnosis'),
 'S02':('NIDDK/NIH','A1C test','https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test'),
 'S03':('NHS','Alzheimer diagnosis','https://www.nhs.uk/conditions/alzheimers-disease/diagnosis/'),
 'S04':('NHLBI/NIH','High blood triglycerides','https://www.nhlbi.nih.gov/health/high-blood-triglycerides'),
 'S05':('NHLBI/NIH','Metabolic syndrome diagnosis','https://www.nhlbi.nih.gov/health/metabolic-syndrome/diagnosis'),
 'S06':('NHLBI/NIH','Blood cholesterol diagnosis','https://www.nhlbi.nih.gov/health/blood-cholesterol/diagnosis'),
 'S07':('NIDDK/NIH','Evaluate CKD','https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/kidney-disease/identify-manage-patients/evaluate-ckd'),
 'S08':('National Kidney Foundation','CKD stages, page2','https://www.kidney.org/sites/default/files/kidneytransplant_updated.pdf'),
 'S09':('NHLBI/NIH','Asthma diagnosis','https://www.nhlbi.nih.gov/health/asthma/diagnosis'),
 'S10':('NIAMS/NIH','Osteoporosis diagnosis','https://www.niams.nih.gov/health-topics/osteoporosis/diagnosis-treatment-and-steps-to-take'),
 'S11':('NHLBI/NIH','Atrial fibrillation diagnosis','https://www.nhlbi.nih.gov/health/atrial-fibrillation/diagnosis'),
 'S12':('NHLBI/NIH','COPD definitions','https://www.nhlbi.nih.gov/health/copd'),
 'S13':('NHLBI/NIH','COPD diagnosis','https://www.nhlbi.nih.gov/health/copd/diagnosis'),
 'S14':('NLM/NIH MedlinePlus Genetics','Alzheimer genetics','https://medlineplus.gov/genetics/condition/alzheimers-disease/'),
 'S15':('KDIGO','2024 CKD guideline, chronicity section','https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf'),
}
# label, symptoms, tests, key distinction, differential, citations, compact note.
ROWS=[
 ('Ischemic heart disease (disorder)','Chest discomfort or exertional breathlessness; symptoms can be absent.','ECG, stress or coronary testing in clinical context.','Risk factors or ordinary vital signs alone do not establish coronary disease.','Other causes of chest symptoms need assessment.',['S01'],'IHD: assess chest symptoms and cardiac testing; routine vitals alone do not establish it.'),
 ('Diabetes mellitus type 2 (disorder)','May lack symptoms.','A1c and plasma glucose.','A1c >=6.5% generally needs confirmation without clear symptoms; glucose tests do not alone determine diabetes type.','Type1 or other diabetes types need distinction.',['S02'],'Diabetes: A1c>=6.5% needs context/confirmation; a glucose test alone does not establish type2.'),
 ("Alzheimer's disease (disorder)",'Changes in memory and daily functioning.','History, cognitive assessment, examination; tests exclude other causes.','A memory score alone is not a complete diagnosis.','Medicines, mood and other health causes can affect cognition.',['S03'],'Alzheimer: assess cognition, function and alternative causes; ordinary labs or one score are insufficient.'),
 ('Hypertriglyceridemia (disorder)','Often no obvious symptoms.','Blood triglyceride measurement.','Use lipid results rather than nonspecific symptoms.','NOT_REVIEWED',['S04'],'High triglycerides: use an actual triglyceride result, not HbA1c or vital signs as substitutes.'),
 ('Metabolic syndrome X (disorder)','Individual components may be asymptomatic.','Waist, blood pressure, glucose, HDL and triglycerides.','A combination of metabolic abnormalities is assessed.','Individual component disorders require distinction.',['S05'],'Metabolic syndrome: assess multiple components including waist, glucose, lipids and BP; not BP alone.'),
 ('Hyperlipidemia (disorder)','Often asymptomatic.','Lipid profile, including LDL and HDL.','A lipid profile is relevant; a normal creatinine does not measure lipids.','NOT_REVIEWED',['S06'],'Hyperlipidemia: requires lipid information; unrelated renal or respiratory measurements do not establish it.'),
 ('Chronic kidney disease stage 4 (disorder)','Symptoms may vary.','eGFR and kidney damage markers, with chronicity.','Stage4 GFR15-29; chronicity must be established.','Acute kidney disease is not automatically chronic.',['S07','S08','S15'],'CKD4: eGFR15-29 plus established chronicity; a single renal result does not prove CKD.'),
 ('Asthma (disorder)','Variable wheezing or breathing symptoms.','Spirometry/PEF and bronchodilator response.','Variable airflow limitation is assessed.','COPD can overlap in symptoms.',['S09'],'Asthma: variable breathing symptoms and lung function/response testing; rate alone is insufficient.'),
 ('Chronic kidney disease stage 2 (disorder)','Early disease may lack symptoms.','eGFR and markers such as urine albumin.','GFR60-89 alone is insufficient; kidney damage and chronicity matter.','Acute versus chronic dysfunction needs distinction.',['S07','S08','S15'],'CKD2: eGFR60-89 alone is not CKD; kidney damage and chronicity must also be established.'),
 ('Chronic kidney disease stage 3 (disorder)','Early disease may lack symptoms.','eGFR, albumin and prior results.','GFR30-59, with chronicity; do not label from one value.','Acute dysfunction can reduce GFR.',['S07','S08','S15'],'CKD3: eGFR30-59 with established chronicity; distinguish acute dysfunction.'),
 ('Osteoporosis (disorder)','Fracture history and risks are relevant.','Bone density DXA, commonly hip/spine.','Bone density and fracture context are important.','NOT_REVIEWED',['S10'],'Osteoporosis: assess DXA and fracture context; normal general labs do not prove or exclude it.'),
 ('Atrial fibrillation (disorder)','Irregular pulse or palpitations may occur.','ECG or rhythm monitoring.','Rhythm recording is important; heart rate alone is insufficient.','Other arrhythmias need distinction.',['S11'],'AF: needs rhythm/ECG assessment; a single heart-rate value is not an AF diagnosis.'),
 ('Chronic obstructive bronchitis (disorder)','Chronic cough with mucus may occur.','COPD assessment and spirometry.','Chronic bronchitis is a COPD component; subtype needs appropriate evidence.','Asthma/other causes of breathing symptoms.',['S12','S13'],'Obstructive bronchitis: chronic cough/mucus and airflow assessment; do not infer subtype from rate alone.'),
 ('Pulmonary emphysema (disorder)','Progressive breathlessness may occur.','COPD evaluation, spirometry and appropriate additional assessment.','Emphysema describes structural air-sac damage.','Chronic bronchitis/asthma can share symptoms.',['S12','S13'],'Emphysema: structural lung damage/COPD assessment; respiratory rate alone does not establish it.'),
 ("Familial Alzheimer's disease of early onset (disorder)",'Cognitive decline beginning earlier in adulthood.','Clinical/family assessment and indicated genetics.','APP/PSEN1/PSEN2 variants can cause familial forms; age or APOE alone is not proof.','Other dementia causes require clinical evaluation.',['S03','S14'],'Familial early Alzheimer: needs clinical/family/genetic context; age or APOE alone is not proof.'),
]

def corpus():
    sources={k:{'publisher':v[0],'title':v[1],'url':v[2],
        'source_version':'2024 guideline' if k=='S15' else 'UPSTREAM_REVISION_NOT_VERIFIED',
        'checked_on':DATE,'verification':'PRIMARY_SOURCE_CONTENT_CHECKED',
        'license':'ORIGINAL_SUMMARY_AND_LINK_ONLY_NO_FULLTEXT_LICENSE_ASSUMED'} for k,v in SOURCES.items()}
    cards=[]
    for i,(label,symptoms,tests,features,differential,ids,note) in enumerate(ROWS,1):
        cards.append({'concept_id':'LOCDI_K%02d'%i,'concept_system':'LOCAL_CATALOG_ID',
            'external_concept_mapping':'NOT_VERIFIED','disease':label,'symptoms':symptoms,'tests':tests,
            'diagnostic_features':features,'differential':differential,
            'differential_status':'NOT_REVIEWED' if differential=='NOT_REVIEWED' else 'SOURCE_SUMMARY_REQUIRES_CLINICAL_REVIEW',
            'source_ids':ids,'common_note':note,'status':'SOURCE_CHECKED_SUMMARY_CLINICAL_REVIEW_NOT_DONE',
            'mapping_scope':'COPD_PARENT_SOURCE_SUBTYPE_NOT_ESTABLISHED' if i in (13,14) else 'CATALOG_TOPIC',
            'independent_clinical_review':False})
    return deepcopy({'version':VERSION,'curated_on':DATE,'sources':sources,'cards':cards})
