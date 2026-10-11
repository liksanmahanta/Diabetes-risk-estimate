"""
Example usage of the private patient records system.
This demonstrates how to interact with patient data with access control.
"""

from patient_records import PatientDatabase, AccessControl, UserRole


def main():
    """Example: Managing patient records with owner access control."""
    
    # Initialize database with owner ID
    owner_id = "liksanmahanta"
    db = PatientDatabase(owner_id)
    
    # Example 1: Owner adds a patient record
    print("=== Adding Patient Record (Owner Only) ===")
    try:
        patient_data = {
            "patient_id": "PAT_12345678",
            "name": "John Doe",
            "date_of_birth": "1985-03-15",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123",
            "medical_history": "Type 2 Diabetes, Hypertension",
            "risk_score": 7.5,
            "assessment_date": "2026-10-11"
        }
        
        result = db.add_patient(owner_id, patient_data)
        print(f"✓ Patient added: {result['patient_id']}")
    except PermissionError as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Retrieve patient record
    print("\n=== Retrieving Patient Record (Owner Only) ===")
    try:
        patient = db.get_patient(owner_id, "PAT_12345678")
        if patient:
            print(f"✓ Patient found:")
            print(f"  Name: {patient['name']}")
            print(f"  Risk Score: {patient['risk_score']}")
            print(f"  Assessment Date: {patient['assessment_date']}")
    except PermissionError as e:
        print(f"✗ Error: {e}")
    
    # Example 3: List all patient records
    print("\n=== Listing All Patients (Owner Only) ===")
    try:
        patients = db.list_patients(owner_id)
        print(f"✓ Found {len(patients)} patient record(s)")
        for p in patients:
            print(f"  - {p['patient_id']}: {p['name']}")
    except PermissionError as e:
        print(f"✗ Error: {e}")
    
    # Example 4: Update patient record
    print("\n=== Updating Patient Record (Owner Only) ===")
    try:
        updates = {
            "risk_score": 8.2,
            "medical_history": "Type 2 Diabetes, Hypertension, High Cholesterol"
        }
        success = db.update_patient(owner_id, "PAT_12345678", updates)
        if success:
            print("✓ Patient record updated")
    except PermissionError as e:
        print(f"✗ Error: {e}")
    
    # Example 5: Attempt access from non-owner user
    print("\n=== Access Control Demo (Non-Owner Denied) ===")
    try:
        other_user = "other_user_123"
        patient = db.get_patient(other_user, "PAT_12345678")
        print(f"✗ Unexpected: Non-owner got access")
    except PermissionError as e:
        print(f"✓ Access correctly denied: {e}")
    
    # Example 6: View access logs
    print("\n=== Access Logs (Owner Only) ===")
    try:
        logs = db.get_access_logs(owner_id, limit=10)
        print(f"✓ Retrieved {len(logs)} log entries:")
        for log in logs[-5:]:  # Show last 5
            print(f"  - {log['timestamp']}: {log['user_id']} - {log['action']} - {log['status']}")
    except PermissionError as e:
        print(f"✗ Error: {e}")
    
    # Example 7: Grant access to clinician
    print("\n=== Managing User Access (Owner Only) ===")
    clinician_id = "clinician_456"
    access_control = db.access_control
    
    if access_control.grant_access(clinician_id, UserRole.CLINICIAN):
        print(f"✓ Granted CLINICIAN access to {clinician_id}")
    
    if access_control.has_access(clinician_id, UserRole.CLINICIAN):
        print(f"✓ Verified {clinician_id} has CLINICIAN access")
    
    # Example 8: Revoke access
    print("\n=== Revoking Access ===")
    if access_control.revoke_access(clinician_id):
        print(f"✓ Revoked access for {clinician_id}")
    
    if not access_control.has_access(clinician_id):
        print(f"✓ Verified {clinician_id} no longer has access")
    
    # Example 9: Delete patient record
    print("\n=== Deleting Patient Record (Owner Only) ===")
    try:
        success = db.delete_patient(owner_id, "PAT_12345678")
        if success:
            print("✓ Patient record deleted")
    except PermissionError as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
