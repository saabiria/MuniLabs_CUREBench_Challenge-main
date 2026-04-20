INTERACTIONS = {

    ("warfarin", "aspirin"): "High bleeding risk",

    ("metformin", "contrast dye"): "Risk of lactic acidosis"
}

def check_interaction(drug1, drug2):

    pair = (drug1.lower(), drug2.lower())

    return INTERACTIONS.get(pair, "No major interaction")