import requests


'''
This function queries the FDA API for adverse events related to a specific drug.

 Args:
    drug_name (str): The name of the drug to query for adverse events.
 Returns:
    str: A string representation of the adverse events related to the drug.
'''
def query_fda(drug_name: str) -> str:
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit=3"
    r = requests.get(url)
    if not r.ok:
        return f"FDA API Error: {r.status_code}"
    results = r.json().get("results", [])
    return str(results[:3])