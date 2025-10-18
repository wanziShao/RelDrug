from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


BASE_DIR = Path('datasets/mimic-iii')
RULES_DIR = Path('FinalRules/mimic-iii/Qwen3-8B')

# 输入文件
KG_INV_FILE = BASE_DIR / 'facts.txt.inv'
ORIGINAL_KG_FILE = BASE_DIR / 'facts.txt'

# 输出和中间文件
NEW_FACTS_FILE = BASE_DIR / 'new_facts.txt'
FINAL_KG_FILE = BASE_DIR / 'kg.txt'

# 规则筛选的超参数
CONFIDENCE_THRESHOLD = 0.9
SUPPORT_THRESHOLD = 50

# 定义允许用于生成新事实的规则头关系
# 这些关系通常是具有明确语义的，可以安全地用于推理
ALLOWED_RULE_HEADS = {
    "may_treat",
    "may_be_treated_by",
    "used_for",
    "has_associated_condition",
    "induces",
    "regulates",
    "has_contraindicated_drug",
    "contraindicated_with_disease",
    "has_contraindicated_class",
    "has_contraindicated_mechanism_of_action",
    "has_contraindicated_physiologic_effect"
}


def load_kg_to_dict(kg_file_path: Path) -> defaultdict:
    """
    将知识图谱文件加载到字典中，便于快速查找。
    结构: {relation: [(head, tail), ...]}
    """
    print(f"正在从 {kg_file_path} 加载知识图谱...")
    kg_dict = defaultdict(list)
    try:
        with open(kg_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载KG"):
                parts = line.strip().split(' ')
                if len(parts) == 3:
                    head, relation, tail = parts
                    kg_dict[relation].append((head, tail))
    except FileNotFoundError:
        print(f"错误: 知识图谱文件 {kg_file_path} 未找到。")
        exit()
    return kg_dict


def apply_rules(rules_dir: Path, kg_dict: defaultdict) -> list:
    """
    遍历规则文件，应用规则以生成新的事实三元组。
    """
    print(f"正在从 {rules_dir} 应用规则...")
    new_facts = []
    rule_files = list(rules_dir.glob('*.txt'))

    if not rule_files:
        print(f"警告: 在目录 {rules_dir} 中未找到任何规则文件 (*.txt)。")
        return []

    for file_path in tqdm(rule_files, desc="处理规则文件"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 4)
                if len(parts) < 5:
                    continue

                # 筛选规则
                support, confidence = float(parts[0]), float(parts[3])
                if confidence < CONFIDENCE_THRESHOLD or support < SUPPORT_THRESHOLD:
                    continue

                rule_paths = parts[4].strip().split(' ', 2)
                rule_head = rule_paths[0]

                if rule_head not in ALLOWED_RULE_HEADS:
                    continue

                relations = [r.strip() for r in rule_paths[2].split(',')]

                # 根据关系路径的长度在KG中寻找匹配并生成新事实
                if len(relations) == 1:
                    # Path: H(X, Y) <= R1(X, Y)
                    rel1 = relations[0]
                    if rel1 in kg_dict:
                        for ent1, ent2 in kg_dict[rel1]:
                            new_facts.append((ent1, rule_head, ent2))

                elif len(relations) == 2:
                    # Path: H(X, Y) <= R1(X, Z), R2(Z, Y)
                    rel1, rel2 = relations
                    if rel1 in kg_dict and rel2 in kg_dict:
                        # 为了提高效率，将第二个关系的实体对创建为查找表
                        rel2_lookup = {h: t for h, t in kg_dict[rel2]}
                        for ent1, ent_mid in kg_dict[rel1]:
                            if ent_mid in rel2_lookup:
                                ent2 = rel2_lookup[ent_mid]
                                new_facts.append((ent1, rule_head, ent2))

                elif len(relations) == 3:
                    # Path: H(X, Y) <= R1(X, Z1), R2(Z1, Z2), R3(Z2, Y)
                    rel1, rel2, rel3 = relations
                    if rel1 in kg_dict and rel2 in kg_dict and rel3 in kg_dict:
                        rel2_lookup = {h: t for h, t in kg_dict[rel2]}
                        rel3_lookup = {h: t for h, t in kg_dict[rel3]}
                        for ent1, z1 in kg_dict[rel1]:
                            if z1 in rel2_lookup:
                                z2 = rel2_lookup[z1]
                                if z2 in rel3_lookup:
                                    ent2 = rel3_lookup[z2]
                                    new_facts.append((ent1, rule_head, ent2))
        print(f"  - 文件 {file_path.name} 处理完毕。")
    return new_facts


def merge_and_deduplicate_kgs(*file_paths: Path, output_path: Path):
    """
    合并多个KG文件，去除重复项，并筛选出关系在 ALLOWED_RULE_HEADS 中的三元组。
    """
    print(f"正在合并文件并写入到 {output_path}...")
    triplets_set = set()

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f"读取 {file_path.name}"):
                    parts = tuple(line.strip().split(' '))
                    if len(parts) == 3:
                        triplets_set.add(parts)
        except FileNotFoundError:
            print(f"警告: 合并时未找到文件 {file_path}，将跳过。")

    # 筛选三元组，只保留关系在 ALLOWED_RULE_HEADS 中的
    filtered_triples = [triple for triple in triplets_set if triple[1] in ALLOWED_RULE_HEADS]
    print(f"合并后共 {len(triplets_set)} 个唯一三元组，筛选后剩余 {len(filtered_triples)} 个。")

    with open(output_path, 'w', encoding='utf-8') as merged_file:
        for triple in tqdm(sorted(list(filtered_triples)), desc="写入最终KG"):
            merged_file.write(' '.join(triple) + '\n')


def main():
    kg_dict = load_kg_to_dict(KG_INV_FILE)
    if not kg_dict:
        print("知识图谱加载失败或为空，任务终止。")
        return

    new_facts = apply_rules(RULES_DIR, kg_dict)
    print(f"\n成功生成 {len(new_facts)} 个新事实。")

    with open(NEW_FACTS_FILE, 'w', encoding='utf-8') as f:
        for fact in tqdm(new_facts, desc="写入新事实"):
            f.write(f"{fact[0]} {fact[1]} {fact[2]}\n")
    print(f"新事实已保存至 {NEW_FACTS_FILE}")

    merge_and_deduplicate_kgs(ORIGINAL_KG_FILE, NEW_FACTS_FILE, output_path=FINAL_KG_FILE)

    print("\n--- 结果统计 ---")
    try:
        with open(FINAL_KG_FILE, 'r', encoding='utf-8') as f:
            final_kg_size = sum(1 for _ in f)
        print(f"原始知识图谱: {ORIGINAL_KG_FILE}")
        print(f"新生成的事实文件: {NEW_FACTS_FILE}")
        print(f"最终合并后的知识图谱: {FINAL_KG_FILE}")
        print(f"最终知识图谱包含的三元组数量: {final_kg_size}")
    except FileNotFoundError:
        print(f"错误: 无法读取最终文件 {FINAL_KG_FILE} 进行统计。")

    print("\n--- 任务全部完成 ---")


if __name__ == "__main__":
    main()

