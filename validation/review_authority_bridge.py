"""Authority interface only. No operational trust registry is configured.

The fake provider is an explicit in-memory TEST_ONLY evidence ledger, not a
cryptographic verifier. Neither caller booleans nor its success admit production.
"""
from copy import deepcopy
from datetime import datetime


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('TIMEZONE_REQUIRED')
    return result


class AuthorityProvider:
    """Future providers must verify issuer/credential/revocation externally.

    This base has no real trust source, public keys, credential authority, or
    institutional authorization. Overriding it is not a configured registry.
    """
    def verify(self, subject_ref, fingerprint, requirements):
        return {'status': 'AUTHENTICATION_NOT_CONFIGURED', 'verified': False,
                'blocker_codes': ['AUTHORITY_UNVERIFIED'], 'production_ready': False}


class TestOnlyAuthority(AuthorityProvider):
    """Separately pinned symbolic grants; never interprets a signature string."""
    def __init__(self, grants, *, verification_at):
        self._grants = deepcopy(grants)
        self._verification_at = verification_at

    def verify(self, subject_ref, fingerprint, requirements):
        codes = []
        row = self._grants.get(subject_ref, {})
        if not row or row.get('purpose') != 'TEST_ONLY' or row.get('subject_fingerprint') != fingerprint:
            codes.append('AUTHORITY_UNVERIFIED')
        for name in ('reviewer_ref', 'axis', 'policy_fingerprint'):
            if row.get(name) != requirements.get(name):
                codes.append('AUTHORITY_SCOPE_MISMATCH')
        for name in ('identity_verified', 'qualification_verified', 'task_suitable', 'independence_verified',
                     'conflict_clear', 'institutional_authority', 'signature_verified', 'evidence_source_verified',
                     'submission_receipt_verified'):
            if row.get(name) is not True:
                codes.append('AUTHORITY_' + name.upper() + '_MISSING')
        if row.get('revoked') is not False:
            codes.append('AUTHORITY_REVOKED_OR_UNKNOWN')
        try:
            at = timestamp(requirements['at'])
            if not timestamp(row['valid_from']) <= at <= timestamp(row['valid_until']):
                codes.append('AUTHORITY_EXPIRED_OR_NOT_YET_VALID')
            if not timestamp(row['valid_from']) <= timestamp(self._verification_at) <= timestamp(row['valid_until']):
                codes.append('AUTHORITY_EXPIRED_OR_NOT_YET_VALID')
        except (KeyError, TypeError, ValueError, AttributeError):
            codes.append('AUTHORITY_VALIDITY_NOT_VERIFIED')
        if requirements.get('requires_blinding') and row.get('blinding_verified') is not True:
            codes.append('BLINDING_NOT_VERIFIED')
        if row.get('submission_sequence') != requirements.get('submission_sequence'):
            codes.append('SUBMISSION_SEQUENCE_NOT_VERIFIED')
        if requirements.get('requires_adjudicator') and row.get('adjudicator_verified') is not True:
            codes.append('ADJUDICATOR_AUTHORITY_UNVERIFIED')
        return {'status': 'TEST_ONLY_VERIFIED' if not codes else 'BLOCKED', 'verified': not codes,
                'blocker_codes': sorted(set(codes)), 'production_ready': False}


def verify_authority(provider, subject_ref, fingerprint, requirements, *, production):
    # Production provider enrollment/issuer verification is deliberately absent.
    # A custom object's verified=True, class flag, or TEST_ONLY ledger cannot opt in.
    if production:
        return {'status': 'AUTHENTICATION_NOT_CONFIGURED', 'verified': False,
                'blocker_codes': ['AUTHORITY_UNVERIFIED'] +
                    (['TEST_ONLY_AUTHORITY_FORBIDDEN_IN_PRODUCTION'] if isinstance(provider, TestOnlyAuthority) else []),
                'production_ready': False}
    if type(provider) is not TestOnlyAuthority:
        return AuthorityProvider().verify(subject_ref, fingerprint, requirements)
    return provider.verify(subject_ref, fingerprint, requirements)
