import dill
from collections import deque, defaultdict
from typing import List, Tuple
import json
from typing import List

def load_entity_mapping(mapping_path: str) -> dict:
    """
    加载 entity_list_mapping.json，返回一个 dict：CUI -> 名称
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

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


with open("../datastes/mimic-iii/voc_final.pkl", "rb") as f:
    voc_data = dill.load(f)

with open("../datastes/mimic-iii/code2cui_voc.pkl", "rb") as f:
    code2cui_data = dill.load(f)


# 根据 type 和 idx，返回对应的 CUI
def id_to_cui(entity_type: str, idx: int) -> str:
    """
    输入：
        entity_type: "diag" / "proc" / "med"
        idx: 在对应 idx2word 中的整数索引
    返回：
        对应的 CUI 字符串。如果找不到则抛出 KeyError 或 返回 None。
    """
    # 根据用户输入的 entity_type 做分支：从不同的 sub-voc 拿 idx2word，再到 code2cui_voc 找 CUI
    if entity_type == "diag":
        code = voc_data["diag_voc"].idx2word.get(idx)
        if code is None:
            raise KeyError(f"diags_voc 中没有找到 idx={idx} 对应的编码。")
        cui = code2cui_data["diag2cui"].code2cui.get(code)
        if cui is None:
            raise KeyError(f"在 code2cui_voc['diag2cui'] 中没有找到诊断编码 {code} 对应的 CUI。")
        return cui

    elif entity_type == "proc":
        code = voc_data["pro_voc"].idx2word.get(idx)
        if code is None:
            raise KeyError(f"pro_voc 中没有找到 idx={idx} 对应的编码。")
        cui = code2cui_data["proc2cui"].code2cui.get(code)
        if cui is None:
            raise KeyError(f"在 code2cui_voc['proc2cui'] 中没有找到手术编码 {code} 对应的 CUI。")
        return cui

    elif entity_type == "med":
        code = voc_data["med_voc"].idx2word.get(idx)
        if code is None:
            raise KeyError(f"med_voc 中没有找到 idx={idx} 对应的编码。")
        cui = code2cui_data["med2cui"].code2cui.get(code)
        if cui is None:
            raise KeyError(f"在 code2cui_voc['med2cui'] 中没有找到药物编码 {code} 对应的 CUI。")
        return cui

    else:
        raise ValueError(f"entity_type 参数只能是 'diag'、'proc' 或 'med'，但收到的是：{entity_type}")


# 从 triples.txt 中提取以给定 CUI 为中心、距离 ≤ 2 跳的所有三元组
def extract_subgraph(center_cui: str, triples_path: str, hop: int = 2) -> List[Tuple[str, str, str]]:
    """
    输入：
        center_cui:   作为中心的实体 CUI（字符串）
        triples_path: triples.txt 的文件路径，每行形如 "head relation tail"
        hop:          距离阈值，返回距离 center_cui 小于等于 hop 跳的三元组（默认2跳）
    返回：
        包含所有“距离 center_cui ≤ hop 跳”的三元组列表，形式为 [(head1, rel1, tail1), ...]
    """
    triples = []
    entity2triples = defaultdict(list)  # 实体 -> 所参与的三元组（无向）

    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
            entity2triples[h].append((h, r, t))
            entity2triples[t].append((h, r, t))

    dist = {center_cui: 0}
    queue = deque([center_cui])

    while queue:
        cur = queue.popleft()
        cur_d = dist[cur]
        if cur_d >= hop:
            continue

        for (h, r, t) in entity2triples.get(cur, []):
            for next_entity in (h, t):
                if next_entity == cur:
                    continue
                if next_entity not in dist:
                    dist[next_entity] = cur_d + 1
                    queue.append(next_entity)

    # 挑出“在 hop 跳范围内”的三元组：要求至少一端在 ≤ hop−1，另一端 ≤ hop
    subgraph_triples = []
    for (h, r, t) in triples:
        d_h = dist.get(h, float('inf'))
        d_t = dist.get(t, float('inf'))
        if min(d_h, d_t) <= hop - 1 and max(d_h, d_t) <= hop:
            subgraph_triples.append((h, r, t))

    return subgraph_triples



if __name__ == "__main__":
    mapping_path = "../datastes/mimic-iii/cui_to_name_map_2hop.json"
    med_names_list = get_med_names_for_indices(mapping_path, id_to_cui, max_idx=130)
    print("med 类型的 idx 从 0 到 130 对应的名称列表：")
    for name in med_names_list:
        print(name)
