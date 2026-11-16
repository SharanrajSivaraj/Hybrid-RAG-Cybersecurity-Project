"""Utility that creates a sample workbook for spreadsheet-loader practice."""

from pathlib import Path

from openpyxl import Workbook


# Provide the create sample workbook behavior used by this module.
def create_sample_workbook() -> None:

    output_path = Path("../data/raw/xlsx/vulnerability_register.xlsx")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workbook = Workbook()

    # First worksheet
    critical_sheet = workbook.active
    critical_sheet.title = "Critical Vulnerabilities"

    critical_sheet.append(
        [
            "CVE ID",
            "Severity",
            "Vulnerability",
            "Affected Asset",
            "Status",
        ]
    )

    critical_sheet.append(
        [
            "CVE-2025-1001",
            "Critical",
            "Remote Code Execution",
            "Production Web Server",
            "Open",
        ]
    )

    critical_sheet.append(
        [
            "CVE-2025-1002",
            "Critical",
            "Authentication Bypass",
            "VPN Gateway",
            "Under Investigation",
        ]
    )

    # Second worksheet
    asset_sheet = workbook.create_sheet("Asset Inventory")

    asset_sheet.append(
        [
            "Asset ID",
            "Asset Name",
            "IP Address",
            "Owner",
            "Environment",
        ]
    )

    asset_sheet.append(
        [
            "AST-001",
            "Production Web Server",
            "10.0.0.10",
            "Infrastructure Team",
            "Production",
        ]
    )

    asset_sheet.append(
        [
            "AST-002",
            "VPN Gateway",
            "10.0.0.20",
            "Network Security Team",
            "Production",
        ]
    )

    workbook.save(output_path)

    print(f"Sample workbook created: {output_path}")


if __name__ == "__main__":
    create_sample_workbook()
