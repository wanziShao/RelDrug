import pickle
import json
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple
from subgraph_extraction import id_to_cui, extract_subgraph
from tqdm import tqdm

def load_records(records_path: str) -> List[Any]:
    """
    加载 records_ori_iv.pkl，返回一个 List，每个元素对应一个患者的就诊记录列表。
    每个患者记录的格式：
    [
        ([diag_id1, diag_id2, ...], [proc_id1, ...], [med_id1, ...]),  # 第一次就诊
        ([...], [...], [...]),  # 第二次就诊
        ...
    ]
    """
    with open(records_path, "rb") as f:
        records = pickle.load(f)
    return records


def load_entity_mapping(mapping_path: str) -> Dict[str, str]:
    """
    加载 entity_list_mapping.json，返回一个 dict：CUI -> 名称
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def get_med_names_for_indices(
    mapping_path: str,
    id_to_cui_fn,
    max_idx: int = 130
) -> List[str]:
    """
    功能：
      - 对于 idx 从 0 到 max_idx（包含 max_idx），调用 id_to_cui_fn("med", idx)
      - 如果成功得到 cui，再从 mapping_path 对应的 JSON 文件中查名称
      - 如果某个 idx 对应的 id_to_cui_fn 抛 KeyError 或者 mapping 中找不到名称，则跳过
      - 返回所有成功映射到名称的 med 名称列表

    参数：
      mapping_path:   entity_list_mapping.json 的路径
      id_to_cui_fn:   用于将 (entity_type, idx) 转 CUI 的函数
      max_idx:        要查询的 med idx 的最大值（默认 130）

    返回：
      List[str]: 将所有有效 idx 对应的 med 名称组成的列表
    """
    # 先加载 CUI -> 名称 的映射
    cui_to_name = load_entity_mapping(mapping_path)

    med_names = []
    for idx in range(0, max_idx + 1):
        try:
            cui = id_to_cui_fn("med", idx)
        except KeyError:
            # 如果该 idx 在词表里查不到对应的 CUI，跳过
            continue

        # 从 JSON 映射里取名称
        name = cui_to_name.get(cui)
        if name:
            med_names.append(name)
        # 如果 mapping 里没有该 CUI，则跳过

    return med_names


def process_patient_records(
        records: List[Any],
        entity_mapping: Dict[str, str],
        triples_path: str,
        hop: int = 2
) -> List[Dict[str, Any]]:
    """
    对所有患者的 records 进行处理。返回一个 List，每个元素是一个字典，包含：
    {
        "visits_named": List[Dict[str, List[str]]],   # 每次就诊的“三个名称列表”
        "merged_graph": Set[Tuple[str, str, str]]     # 该患者所有 visit 对应子图三元组合并后的集合
    }

    参数：
        records:         load_records(...) 得到的列表，按患者分组，结构如题描述。
        entity_mapping:  load_entity_mapping(...) 得到的 CUI->名称 映射字典
        triples_path:    triples.txt 的文件路径
        hop:             extract_subgraph 中使用的跳数阈值，默认为 2
    """
    all_patients_output: List[Dict[str, Any]] = []

    for patient_idx, patient_record in tqdm(enumerate(records)):
        # patient_record: List of visits；每个 visit 是 ([diag_ids], [proc_ids], [med_ids])
        visits_named: List[Dict[str, List[str]]] = []
        merged_graph: Set[Tuple[str, str, str]] = set()

        for visit in patient_record:
            diag_ids, proc_ids, med_ids = visit  # 解包

            # 1. ID -> CUI -> 名称
            diag_cuis = []
            for d_id in diag_ids:
                try:
                    d_cui = id_to_cui("diag", d_id)
                except KeyError:
                    # 如果没找到就跳过
                    continue
                diag_cuis.append(d_cui)

            proc_cuis = []
            for p_id in proc_ids:
                try:
                    p_cui = id_to_cui("proc", p_id)
                except KeyError:
                    continue
                proc_cuis.append(p_cui)

            med_cuis = []
            for m_id in med_ids:
                try:
                    m_cui = id_to_cui("med", m_id)
                except KeyError:
                    continue
                med_cuis.append(m_cui)

            # CUI -> 名称（如果 mapping 中没找到名称，就保留 CUI）
            diag_names = [entity_mapping.get(cui, cui) for cui in diag_cuis]
            proc_names = [entity_mapping.get(cui, cui) for cui in proc_cuis]
            med_names = [entity_mapping.get(cui, cui) for cui in med_cuis]

            visits_named.append({
                "diagnosis": diag_names,
                "procedure": proc_names,
                "medication": med_names
            })

            # 2. 利用 extract_subgraph 生成以每个 CUI 为中心的 hop 跳子图，把所有三元组并入 merged_graph
            for center_cui in diag_cuis + proc_cuis + med_cuis:
                try:
                    sub_triples = extract_subgraph(center_cui, triples_path, hop=hop)
                except Exception:
                    # 如果 extract_subgraph 运行失败，可以选择跳过或记录日志
                    continue

                for tri in sub_triples:
                    merged_graph.add(tri)

        # 将该患者结果加入总列表
        all_patients_output.append({
            "visits_named": visits_named,
            "merged_graph": merged_graph
        })

    return all_patients_output
