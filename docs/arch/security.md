---
title: Security Guide
---

# Reporting Security Issues

The Apache Software Foundation takes a very active stance in eliminating
security problems and denial of service attacks against its products. We
strongly encourage folks to report such problems to our private security
mailing list first, before disclosing them in a public forum.

Please note that the security mailing list should only be used for
reporting undisclosed security vulnerabilities and managing the process
of fixing such vulnerabilities. We cannot accept regular bug reports or
other queries at this address. All mail sent to this address that does
not relate to an undisclosed security problem in our source code will be
ignored. Questions about: if a vulnerability applies to your particular
application obtaining further information on a published vulnerability
availability of patches and/or new releases should be addressed to to
the user discuss forum.

The private security mailing address is:
[security@apache.org](mailto:security@apache.org). Feel free to consult the
[Apache Security guide](https://www.apache.org/security/).

# Considerations

The default binary generated by TVM only relies on a minimum runtime
API. The runtime depends on a limited set of system calls(e.g. malloc)
in the system library.

AutoTVM data exchange between the tracker, server and client are in
plain-text. It is recommended to use them under trusted networking
environment or encrypted channels.