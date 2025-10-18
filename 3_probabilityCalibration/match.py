# 更新可能性概率
import json
import re
from typing import Dict, List, Tuple
import dill

def step_one_build_id_to_name(
    voc_path: str,
    code2cui_path: str,
    entity_mapping_path: str
) -> Dict[int, str]:
    """
    Build a mapping from drug ID to drug name.
    - Traverse voc_iv_sym1_mulvisit.pkl to get idx -> word.
    - Use code2cui_voc.pkl to map word -> cui.
    - Use entity_list_mapping.json to map cui -> name.
    Returns a dictionary: id (int) -> name (str).
    """
    with open(voc_path, "rb") as f:
        voc_data = dill.load(f)
    med_idx2word: Dict[int, str] = voc_data["med_voc"].idx2word

    with open(code2cui_path, "rb") as f:
        code2cui_data = dill.load(f)
    med_code2cui: Dict[str, str] = code2cui_data["med2cui"].code2cui

    with open(entity_mapping_path, "r", encoding="utf-8") as f:
        entity_mapping: Dict[str, str] = json.load(f)

    id_to_name: Dict[int, str] = {}
    for idx, code in med_idx2word.items():
        cui = med_code2cui.get(code)
        if cui is None:
            continue
        name = entity_mapping.get(cui)
        if name is None:
            continue
        id_to_name[idx] = name
    return id_to_name

def parse_recommendation(recommendation: str) -> List[Tuple[str, float]]:
    """
    Parse the recommendation string to extract drug names and probabilities.
    """
    pattern = r'\(\s*"([^"]+)"\s*,\s*"(\d+\.\d+)"\s*\)'
    matches = re.findall(pattern, recommendation)
    parsed = []
    for drug_name, prob_str in matches:
        try:
            prob = float(prob_str)
            parsed.append((drug_name, prob))
        except ValueError:
            print(f"Warning: Invalid probability '{prob_str}' for drug '{drug_name}'")
    return parsed

def build_name_to_id(id_to_name: Dict[int, str]) -> Dict[str, int]:
    """
    Reverse the id_to_name dictionary to get name_to_id.
    Handles duplicate names by keeping the first occurrence.
    """
    name_to_id = {}
    for idx, name in id_to_name.items():
        if name not in name_to_id:
            name_to_id[name] = idx
        else:
            print(f"Warning: Duplicate name '{name}' found; using ID {name_to_id[name]}")
    return name_to_id

def update_predicted(predicted: List[List[float]], update_dict: Dict[int, float]) -> List[List[float]]:
    """
    Update the predicted probabilities based on the update dictionary.
    If a med_id exists in update_dict, update its probability; otherwise, keep it unchanged.
    """
    updated = []
    for med_id, prob in predicted:
        if med_id in update_dict:
            updated.append([med_id, update_dict[med_id]])
        else:
            updated.append([med_id, prob])
    return updated

def main():
    prob_json_path = "../2_deepModelDrugRec/saved_results/mimic-iii/IntentKG_final/rec-real-prob-all.json"
    voc_path = "../datastes/mimic-iii/voc_final.pkl"
    code2cui_path = "../datastes/mimic-iii/code2cui_voc.pkl"
    entity_mapping_path = "../datastes/mimic-iii/cui_to_name_map_2hop.json"
    v28_json_path = "../datastes/mimic-iii/refined_subgraph.json"
    output_path = "../datastes/mimic-iii/undata_refined_subgraph.json"


    id_to_name = step_one_build_id_to_name(voc_path, code2cui_path, entity_mapping_path)
    name_to_id = build_name_to_id(id_to_name)

    with open(v28_json_path, 'r', encoding='utf-8') as f:
        v28_data = json.load(f)

    updates = {}  # (patient_id, visit_id) -> {med_id: prob}
    for entry in v28_data:
        patient_id = entry['patient_index']
        visit_id = entry['visit_index']
        recommendation = entry['recommendation']
        parsed = parse_recommendation(recommendation)
        update_dict = {}
        for drug_name, prob in parsed:
            med_id = name_to_id.get(drug_name)
            if med_id is not None:
                update_dict[med_id] = prob
        if update_dict:
            updates[(patient_id, visit_id)] = update_dict

    with open(prob_json_path, 'r', encoding='utf-8') as f:
        prob_data = json.load(f)

    for patient in prob_data:
        patient_id = patient['patient_id']
        for visit in patient['visits']:
            visit_id = visit['visit_id']
            key = (patient_id, visit_id)
            if key in updates:
                visit['predicted'] = update_predicted(visit['predicted'], updates[key])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prob_data, f, ensure_ascii=False, indent=2)
    print(f"Updated data saved to {output_path}")

if __name__ == "__main__":
    main()