import json
import requests
import os


def biocyc_credentials(dir_credentials):
    s = requests.Session()
    cred_path = os.path.join(dir_credentials, "biocyc_credentials.json")
    with open(cred_path, "r") as f:
        credentials = json.load(f)

    s.post(
        "https://websvc.biocyc.org/credentials/login/",
        data={"email": credentials["email"], "password": credentials["password"]},
    )
    return s
