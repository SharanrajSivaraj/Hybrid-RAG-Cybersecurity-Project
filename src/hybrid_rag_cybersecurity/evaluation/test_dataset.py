"""Question, expected-source, and expected-keyword evaluation cases."""

EVALUATION_DATASET = [
    {
        "id": "Q001",
        "question": "What are the seven steps in the incident response process?",
        "expected_sources": ["incident_response_policy.txt"],
        "expected_keywords": [
            "detection",
            "reporting",
            "triage",
            "containment",
            "eradication",
            "recovery",
            "lessons learned",
        ],
        "answerable": True,
    },
    {
        "id": "Q002",
        "question": "What is the initial triage target for a critical incident?",
        "expected_sources": ["incident_response_policy.txt"],
        "expected_keywords": ["15 minutes"],
        "answerable": True,
    },
    {
        "id": "Q003",
        "question": "What is the severity and vulnerability type of CVE-2025-1001?",
        "expected_sources": ["vulnerability_register.xlsx"],
        "expected_keywords": ["critical", "remote code execution"],
        "answerable": True,
    },
    {
        "id": "Q004",
        "question": "Which asset is associated with CVE-2025-1001?",
        "expected_sources": ["vulnerability_register.xlsx"],
        "expected_keywords": ["production web server"],
        "answerable": True,
    },
    {
        "id": "Q005",
        "question": ("What are the NIST CSF 2.0 Respond and Recover functions about?"),
        "expected_sources": ["NIST.CSWP.29.pdf"],
        "expected_keywords": [
            "incident management",
            "incident analysis",
            "incident response reporting and communication",
            "incident mitigation",
            "incident recovery plan execution",
            "incident recovery communication",
        ],
        "answerable": True,
        "retrieval_queries": {
            "Respond": (
                "NIST CSF 2.0 RESPOND RS: "
                "Incident Management RS.MA, "
                "Incident Analysis RS.AN, "
                "Incident Response Reporting and Communication RS.CO, "
                "Incident Mitigation RS.MI. "
                "What cybersecurity outcomes are defined?"
            ),
            "Recover": (
                "NIST CSF 2.0 RECOVER RC: "
                "Incident Recovery Plan Execution RC.RP, "
                "Incident Recovery Communication RC.CO. "
                "What cybersecurity outcomes are defined?"
            ),
        },
    },
    {
        "id": "Q006",
        "question": "What is the company's cyber insurance policy number?",
        "expected_sources": [],
        "expected_keywords": [],
        "answerable": False,
    },
]
