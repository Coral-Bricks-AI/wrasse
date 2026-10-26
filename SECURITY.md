# Security

## Reporting

Report vulnerabilities privately through GitHub's "Report a vulnerability"
(Security → Advisories) on this repository. Please do not open a public issue
for anything exploitable.

Useful reports name the invariant that breaks. The most important one is in
`docs/trust-model.md`: **the review must never execute code from the pull
request**. Anything that lets a PR cause code execution, exfiltrate the
provider key, or post content the harness did not generate is in scope.

## Supported versions

The latest minor release. The action pins to a major tag (`@v1`), so security
fixes reach users of that tag without a workflow change.
