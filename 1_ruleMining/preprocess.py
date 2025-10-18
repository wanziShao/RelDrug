from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path('datasets/mimic-iii')

# 输入文件
RELATION_LIST_FILE = BASE_DIR / 'relation_list.txt'
KG_FILE = BASE_DIR / 'facts.txt'

# 输出文件
TRUE_RELATIONS_FILE = BASE_DIR / 'true_relations.txt'
ENTITIES_FILE = BASE_DIR / 'entities.txt'
RELATIONS_FILE = BASE_DIR / 'relations.txt'
KG_INV_FILE = BASE_DIR / 'facts.txt.inv'


def process_true_relations(input_file, output_file):
    """
    处理 relation_list.txt，跳过首行，并交换前两列后写入 true_relations.txt。
    """
    print(f"开始处理 {input_file.name}...")
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()[1:]  # 跳过标题行

        with open(output_file, 'w') as outfile:
            for line in tqdm(lines, desc="处理 relation_list"):
                parts = line.split()
                if len(parts) >= 2:
                    outfile.write(f"{parts[1]} {parts[0]}\n")

        print(f"处理完成，结果已写入 {output_file.name}")
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到。")
    except Exception as e:
        print(f"处理 {input_file.name} 时发生错误: {e}")


def extract_entities(input_file, output_file):
    """
    从 kg.txt 提取所有唯一的实体（头实体和尾实体），排序后写入 entities.txt。
    """
    print(f"\n开始从 {input_file.name} 提取实体...")
    entity_set = set()
    try:
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="提取实体"):
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    entity_set.add(parts[0])  # 头实体
                    entity_set.add(parts[2])  # 尾实体

        entity_list = sorted(list(entity_set))
        with open(output_file, 'w') as f:
            for entity in tqdm(entity_list, desc="写入实体"):
                f.write(entity + '\n')

        print(f"实体提取完成，已写入 {output_file.name}")
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到。")
    except Exception as e:
        print(f"提取实体时发生错误: {e}")


def extract_relations(input_file, output_file):
    """
    从 kg.txt 提取所有唯一的关系，排序后写入 relations.txt。
    """
    print(f"\n开始从 {input_file.name} 提取关系...")
    relation_set = set()
    try:
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="提取关系"):
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    relation_set.add(parts[1])  # 关系

        relation_list = sorted(list(relation_set))
        with open(output_file, 'w') as f:
            for relation in tqdm(relation_list, desc="写入关系"):
                f.write(relation + '\n')

        print(f"关系提取完成，已写入 {output_file.name}")
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到。")
    except Exception as e:
        print(f"提取关系时发生错误: {e}")


def create_inverse_kg(input_file, output_file):
    """
    读取 kg.txt，为每个三元组 (h, r, t) 添加一个反向三元组 (t, inv_r, h)，
    并将两者都写入 kg.txt.inv。
    """
    print(f"\n开始创建反向知识图谱 {output_file.name}...")
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in tqdm(f_in, desc="创建反向KG"):
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    head = parts[0]
                    relation = parts[1]
                    tail = parts[2]

                    f_out.write(' '.join([head, relation, tail]) + '\n')

                    relation_inv = 'inv_' + relation
                    f_out.write(' '.join([tail, relation_inv, head]) + '\n')

        print(f"反向知识图谱创建完成，已写入 {output_file.name}")
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到。")
    except Exception as e:
        print(f"创建反向KG时发生错误: {e}")


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    process_true_relations(RELATION_LIST_FILE, TRUE_RELATIONS_FILE)
    extract_entities(KG_FILE, ENTITIES_FILE)
    extract_relations(KG_FILE, RELATIONS_FILE)
    create_inverse_kg(KG_FILE, KG_INV_FILE)
    print("\n--- 所有数据处理步骤已全部完成 ---")

if __name__ == "__main__":
    main()