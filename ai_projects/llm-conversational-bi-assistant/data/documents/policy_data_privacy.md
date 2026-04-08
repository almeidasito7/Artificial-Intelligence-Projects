# Data Privacy & Security Policy

**Document ID:** POL-DPS-001  
**Effective Date:** March 1, 2025  
**Last Reviewed:** December 15, 2025  
**Owner:** Information Security

## Scope

This policy applies to all employees, contractors, and third-party vendors who access, process, or store personal data within our systems. Personal data includes candidate information, contractor records, client data, and any information that can identify an individual.

## Data Classification

All data is classified into four tiers:

- **Public:** Information approved for external publication (job postings, marketing materials).
- **Internal:** Information for internal use only (SOPs, internal reports, aggregated analytics).
- **Confidential:** Sensitive business data (client contracts, bill rates, margin data, revenue reports).
- **Restricted:** Personally identifiable information (SSN, bank account details, medical records, background check results).

## Access Control

Access to data is governed by the principle of least privilege:

- Users are granted access only to the data necessary for their role.
- Regional analysts can only view data for their assigned regions and divisions.
- Access is enforced at the application and database level through Row-Level Security (RLS).
- All access grants must be approved by the data owner and reviewed quarterly.
- Privileged access (admin, database direct access) requires MFA and is logged.

## Data Handling Rules

**Confidential and Restricted Data:**
- Must not be shared via email without encryption.
- Must not be stored on local devices or personal cloud storage.
- Must not be included in LLM prompts or AI tool inputs unless the tool has been approved by Information Security and implements appropriate data isolation.
- Must be masked or redacted in demo environments and training materials.

**AI & LLM Usage:**
- Approved AI tools must implement Row-Level Security to prevent unauthorized data exposure.
- LLM-generated responses based on confidential data must not be cached in shared caches accessible to users with different permission levels.
- All AI tool interactions involving confidential data must be logged for audit purposes.
- Candidate PII (names, contact info, SSN) must never be sent to external LLM APIs. Use anonymized or aggregated data instead.

## Data Retention

- Active placement records: Retained for the duration of the assignment plus 3 years.
- Candidate profiles: Retained for 5 years from last activity date, then anonymized.
- Client contracts: Retained for 7 years after contract expiration.
- Timesheets and payroll records: Retained for 7 years per IRS requirements.
- Background check results: Destroyed after 2 years per Fair Credit Reporting Act guidelines.

## Incident Response

Data breaches or suspected unauthorized access must be reported to the Information Security team within 1 hour of discovery. The incident response process follows four phases:

1. **Identification:** Confirm the breach and assess scope.
2. **Containment:** Isolate affected systems, revoke compromised credentials.
3. **Notification:** Notify affected individuals within 72 hours. Notify regulatory authorities as required by state breach notification laws.
4. **Remediation:** Implement fixes, update access controls, document lessons learned.

## Compliance

This policy aligns with the following regulations:
- California Consumer Privacy Act (CCPA)
- Health Insurance Portability and Accountability Act (HIPAA) — for healthcare placements
- General Data Protection Regulation (GDPR) — for international operations
- SOC 2 Type II controls

Annual compliance audits are conducted by an external firm. All employees complete data privacy training during onboarding and annually thereafter.
