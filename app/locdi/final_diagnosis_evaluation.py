"""FINAL-EXP post-hoc Source agreement. No inferred clinical equivalence."""
import re
import unicodedata
from app.locdi.llm_setup import digest


def normalized(value):
    if not isinstance(value, str):
        return None
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold()))


def diagnosis_match(prediction, references, *, certainty='supported', equivalences=()):
    """Original string score is retained; synonym rules require pinned evidence.

    Different strings without a reviewed code/synonym rule remain REVIEW_REQUIRED.
    Related diseases, same specialty and numerical text similarity are never rules.
    """
    if not isinstance(references, list) or not references or any(not isinstance(x, str) for x in references):
        raise ValueError('EXPLICIT_SOURCE_REFERENCES_REQUIRED')
    original = prediction in references if isinstance(prediction, str) else False
    if certainty != 'supported' or not isinstance(prediction, str) or not prediction.strip():
        return {'original_exact': original, 'normalized_exact': False,
                'concept_match': None, 'classification': 'UNRESOLVED', 'rule_id': None}
    norm = normalized(prediction)
    if norm in [normalized(x) for x in references]:
        return {'original_exact': original, 'normalized_exact': True,
                'concept_match': True, 'classification': 'NORMALIZED_EXACT',
                'rule_id': 'NFKC_CASE_PUNCTUATION_WHITESPACE.v1'}
    for rule in equivalences:
        if (set(rule) != {'terms', 'evidence_sha256', 'evidence_path', 'relation', 'review_status'}
                or rule['relation'] not in ('SAME_STANDARD_CODE', 'VERIFIED_SYNONYM')
                or rule['review_status'] != 'VERIFIED'
                or not re.fullmatch('[0-9a-f]{64}', rule['evidence_sha256'])
                or not rule['evidence_path']):
            raise ValueError('UNVERIFIED_EQUIVALENCE_RULE')
        terms = {normalized(x) for x in rule['terms']}
        if norm in terms and any(normalized(x) in terms for x in references):
            return {'original_exact': original, 'normalized_exact': False,
                    'concept_match': True, 'classification': 'EVIDENCED_EQUIVALENT',
                    'rule_id': digest(rule)}
    return {'original_exact': original, 'normalized_exact': False,
            'concept_match': None, 'classification': 'REVIEW_REQUIRED', 'rule_id': None}
