import json
import dill
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from typing import List, Tuple

# --- 1. 配置区域 ---

# 输入文件路径
JSON_FILE_PATH = '../datastes/mimic-iii/undata_refined_subgraph.json'
DDI_MATRIX_PATH = '../datastes/mimic-iii/ddi_A_final.pkl'

PROBABILITY_THRESHOLD = 0.5


def sequence_metric(y_gt: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, y_label: np.ndarray) -> Tuple[float, ...]:
    """
    计算一套完整的序列推荐指标。

    Args:
        y_gt: 真实标签 (ground truth), shape (num_visits, num_medications)
        y_pred: 预测标签 (predicted labels, prob > threshold), shape (num_visits, num_medications)
        y_prob: 预测概率 (predicted probabilities), shape (num_visits, num_medications)
        y_label: 按概率排序的预测药物ID列表, shape (num_visits, num_predicted_meds)

    Returns:
        一个包含多个评估指标的元组。
    """

    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(avg_prc, avg_recall):
        score = []
        for idx in range(len(avg_prc)):
            if (avg_prc[idx] + avg_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2 * avg_prc[idx] * avg_recall[idx] / (avg_prc[idx] + avg_recall[idx]))
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = [f1_score(y_gt[b], y_pred[b], average='macro') for b in range(y_gt.shape[0])]
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = [roc_auc_score(y_gt[b], y_pred_prob[b], average='macro') for b in range(len(y_gt))]
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = [average_precision_score(y_gt[b], y_prob[b], average='macro') for b in range(len(y_gt))]
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            # 确保 y_prob_label[i] 的长度至少为 k
            # 如果不够长，则在现有预测中计算
            limit = min(k, len(y_prob_label[i]))
            if limit == 0: continue

            for j in y_prob_label[i][:limit]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    # --- 指标计算 ---
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0

    ja = jaccard(y_gt, y_label)
    prauc = precision_auc(y_gt, y_prob)

    avg_prc_val = average_prc(y_gt, y_label)
    avg_recall_val = average_recall(y_gt, y_label)
    avg_f1_val = average_f1(avg_prc_val, avg_recall_val)

    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)

    return ja, prauc, np.mean(avg_prc_val), np.mean(avg_recall_val), np.mean(avg_f1_val), p_1, p_3, p_5


def ddi_rate_score(record: List[List[List[int]]], ddi_matrix: np.ndarray) -> float:
    """
    计算给定记录的 DDI (药物-药物相互作用) 率。

    Args:
        record: 包含多位患者的预测药物列表。
                格式: [[ [visit1_meds], [visit2_meds] ], [ [patient2_visit1_meds] ]]
        ddi_matrix: 一个 N x N 的邻接矩阵，ddi_matrix[i, j] == 1 表示药物i和j有相互作用。

    Returns:
        DDI 率。
    """
    all_pairs_count = 0
    ddi_count = 0
    for patient_visits in record:
        for med_set in patient_visits:
            # 检查该次就诊内的药物两两组合
            for i, med1 in enumerate(med_set):
                for med2 in med_set[i + 1:]:
                    all_pairs_count += 1
                    if ddi_matrix[med1, med2] == 1 or ddi_matrix[med2, med1] == 1:
                        ddi_count += 1

    return (ddi_count / all_pairs_count) if all_pairs_count > 0 else 0.0


def main():
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功从 {JSON_FILE_PATH} 加载数据。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 加载 JSON 文件失败 ({JSON_FILE_PATH}): {e}")
        exit(1)

    try:
        ddi_matrix = dill.load(open(DDI_MATRIX_PATH, 'rb'))
        print(f"成功从 {DDI_MATRIX_PATH} 加载 DDI 矩阵。")
    except FileNotFoundError:
        print(f"错误: DDI 矩阵文件未找到: {DDI_MATRIX_PATH}")
        exit(1)

    all_med_ids = set()
    for patient in data:
        for visit in patient.get('visits', []):
            all_med_ids.update(visit.get('actual', []))
            for med_id, _ in visit.get('predicted', []):
                all_med_ids.add(med_id)
    if not all_med_ids:
        print("错误: 数据中未找到任何药物信息。")
        exit(1)
    num_medications = max(all_med_ids) + 1

    all_y_gt, all_y_pred, all_y_prob, all_y_label = [], [], [], []
    ddi_record = []
    visit_pred_counts = []

    print("正在处理所有患者的就诊数据...")
    for patient in data:
        patient_med_sets_for_ddi = []
        for visit in patient.get('visits', []):
            y_gt = np.zeros(num_medications, dtype=int)
            y_gt[visit.get('actual', [])] = 1

            predicted_meds = visit.get('predicted', [])

            y_prob = np.zeros(num_medications, dtype=float)
            for med_id, prob in predicted_meds:
                y_prob[med_id] = prob

            positive_preds = {med_id: prob for med_id, prob in predicted_meds if prob > PROBABILITY_THRESHOLD}

            y_pred = np.zeros(num_medications, dtype=int)
            if positive_preds:
                y_pred[list(positive_preds.keys())] = 1

            sorted_positive_preds = sorted(positive_preds.items(), key=lambda x: x[1], reverse=True)
            y_label_visit = [med_id for med_id, _ in sorted_positive_preds]

            all_y_gt.append(y_gt)
            all_y_pred.append(y_pred)
            all_y_prob.append(y_prob)
            all_y_label.append(y_label_visit)

            meds_for_ddi = list(positive_preds.keys())
            patient_med_sets_for_ddi.append(meds_for_ddi)

            visit_pred_counts.append(len(positive_preds))

        ddi_record.append(patient_med_sets_for_ddi)

    all_y_gt = np.array(all_y_gt)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    print("\n" + "=" * 30)
    print("      评 估 结 果")
    print("=" * 30)

    ja, prauc, avg_prc, avg_recall, avg_f1, p1, p3, p5 = sequence_metric(
        all_y_gt, all_y_pred, all_y_prob, all_y_label
    )
    print("\n--- 推荐系统指标 ---")
    print(f"Jaccard:                 {ja:.4f}")
    print(f"PRAUC:                   {prauc:.4f}")
    print(f"Average Precision:       {avg_prc:.4f}")
    print(f"Average Recall:          {avg_recall:.4f}")
    print(f"Average F1-score:        {avg_f1:.4f}")
    print(f"Precision@1:             {p1:.4f}")
    print(f"Precision@3:             {p3:.4f}")
    print(f"Precision@5:             {p5:.4f}")

    ddi_rate = ddi_rate_score(ddi_record, ddi_matrix)
    print("\n--- 药物安全性指标 ---")
    print(f"DDI Rate:                {ddi_rate:.4f}")

    avg_pred_count = np.mean(visit_pred_counts) if visit_pred_counts else 0
    print("\n--- 其他统计 ---")
    print(f"平均每次就诊推荐药物数:  {avg_pred_count:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
