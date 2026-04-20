DRUG_DATABASE = {

    "warfarin": {
        "class": "anticoagulant",
        "renal_adjustment": False
    },

    "metformin": {
        "class": "antidiabetic",
        "renal_adjustment": True
    }
}

def lookup_drug(drug_name):

    drug_name = drug_name.lower()

    return DRUG_DATABASE.get(drug_name, "Drug not found")