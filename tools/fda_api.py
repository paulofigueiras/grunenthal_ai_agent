import requests

def query_fda(drug_name: str) -> str:
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit=3"
    r = requests.get(url)
    if not r.ok:
        return f"FDA API Error: {r.status_code}"
    results = r.json().get("results", [])
    return str(results[:3])