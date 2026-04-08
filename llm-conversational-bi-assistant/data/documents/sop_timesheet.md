# Standard Operating Procedure: Timesheet Processing

**Document ID:** SOP-TS-002  
**Effective Date:** February 1, 2025  
**Last Reviewed:** February 15, 2026  
**Owner:** Payroll & Billing Operations

## Purpose

This document defines the end-to-end timesheet processing workflow, from contractor submission through client billing and contractor payment.

## Submission Deadlines

Contractors must submit timesheets weekly by **Monday at 12:00 PM EST** for the prior work week (Monday through Sunday). Late submissions delay payment processing and may result in the contractor being paid in the following pay cycle.

For clients on bi-weekly billing cycles, timesheets are aggregated and invoiced on the 1st and 15th of each month.

## Approval Workflow

1. **Contractor submits** timesheet through the client's VMS platform.
2. **Client supervisor approves** the timesheet within the VMS. The standard approval SLA is 48 hours from submission.
3. **Our system (Dormammu pipeline)** automatically pulls approved timesheets from the VMS via API integration.
4. **Payroll coordinator reviews** for discrepancies: overtime calculations, rate mismatches, or missing entries.
5. **Payroll processes** contractor payment via direct deposit. Standard pay cycle is weekly, with payments issued every Friday.

## Rate Validation

During processing, the system validates that:

- The billed hours do not exceed the client-approved maximum (typically 40 hours/week unless overtime is pre-approved)
- The bill rate and pay rate match the placement agreement in Job Diva
- Overtime rates are correctly calculated at 1.5x the standard pay rate
- Any special rates (holiday pay, shift differentials) are applied correctly

Discrepancies are flagged for manual review. The payroll coordinator must resolve all flags before payment processing.

## Overtime Policy

Overtime must be pre-approved by the client supervisor. Contractors are not authorized to work overtime without written approval. Overtime is defined as:

- Hours exceeding 40 in a work week (federal standard)
- Hours exceeding 8 in a work day (California and select states)
- Any work on designated company holidays

Bill rate for overtime is calculated at 1.5x the standard bill rate. Pay rate follows the same multiplier unless a different overtime structure is defined in the placement agreement.

## Error Handling

If a timesheet contains errors:

- **Underbilled hours:** A correction timesheet is submitted for the difference in the following period.
- **Overbilled hours:** A credit memo is issued to the client, and the contractor's next paycheck is adjusted accordingly.
- **Rate errors:** The payroll coordinator escalates to the account manager for client communication and rate correction.

All corrections must be documented in the placement file and communicated to both the contractor and client within 2 business days.

## VMS Integration Status

| VMS Platform   | Integration Type | Auto-Pull | Status     |
|----------------|-----------------|-----------|------------|
| Fieldglass     | API (REST)      | Yes       | Production |
| Beeline        | API (REST)      | Yes       | Production |
| VNDLY          | SFTP            | Yes       | Production |
| Coupa          | API (REST)      | Yes       | Beta       |
| IQNavigator    | Manual Export   | No        | Manual     |

For IQNavigator clients, timesheets must be manually exported and uploaded to our system weekly. This process is targeted for automation in Q3 2026.

## Escalation

Timesheet processing issues that cannot be resolved within 48 hours are escalated to the Regional Operations Manager. Client-side approval delays exceeding 5 business days are escalated to the Account Director for client intervention.
