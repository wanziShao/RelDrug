import torch
import torch.nn as nn

def mimic_data(data):
    """
    转换数据为DataLoader可用的格式，增加病人的就诊记录数量
    """
    min_visits = 2
    disease = []
    procedure = []
    medication = []

    for patient in data:
        disease_patient = []
        procedure_patient = []
        medication_patient = []

        # 如果病人的记录少于min_visits，则进行增强（复制）
        while len(patient) < min_visits:
            patient.append(patient[0])  # 简单复制病人的第一条记录（你可以根据需要改成更复杂的增强方式）

        for admission in patient:
            disease_patient.append(admission[0])  # diagnosis codes
            procedure_patient.append(admission[1])  # procedure codes
            medication_patient.append(admission[2])  # medication codes

        disease.append(disease_patient)
        procedure.append(procedure_patient)
        medication.append(medication_patient)

    return list(zip(disease, procedure, medication))


def pad_batch_v2_train(batch):
    """
    对训练数据进行padding和转换
    返回19个值,包含length矩阵
    """
    max_visit = max([len(p[0]) for p in batch])  # 最大就诊次数
    max_diag_len = max([len(d) for p in batch for d in p[0]])  # 最大诊断代码长度
    max_proc_len = max([len(d) for p in batch for d in p[1]])  # 最大手术代码长度
    max_med_len = max([len(d) for p in batch for d in p[2]])  # 最大药物代码长度

    batch_disease = []
    batch_proc = []
    batch_med = []
    batch_mask_d = []
    batch_mask_p = []
    batch_mask_m = []

    # 增加length矩阵
    batch_d_length = []
    batch_p_length = []
    batch_m_length = []

    batch_dec_disease = []
    batch_stay_disease = []
    batch_dec_disease_mask = []
    batch_stay_disease_mask = []

    batch_dec_proc = []
    batch_stay_proc = []
    batch_dec_proc_mask = []
    batch_stay_proc_mask = []

    batch_target = []

    for patient in batch:
        disease_list = []
        proc_list = []
        med_list = []
        d_mask_list = []
        p_mask_list = []
        m_mask_list = []

        # 增加length list
        d_length_list = []
        p_length_list = []
        m_length_list = []

        dec_disease_list = []
        stay_disease_list = []
        dec_disease_mask_list = []
        stay_disease_mask_list = []

        dec_proc_list = []
        stay_proc_list = []
        dec_proc_mask_list = []
        stay_proc_mask_list = []

        target_list = []

        for idx, visit in enumerate(patient[0]):
            # padding for diagnosis
            diagnosis = visit + [-1] * (max_diag_len - len(visit))
            disease_list.append(diagnosis)
            d_mask_list.append([1] * len(visit) + [0] * (max_diag_len - len(visit)))
            d_length_list.append(len(visit))

            # padding for procedure
            proc = patient[1][idx] + [-1] * (max_proc_len - len(patient[1][idx]))
            proc_list.append(proc)
            p_mask_list.append([1] * len(patient[1][idx]) + [0] * (max_proc_len - len(patient[1][idx])))

            # padding for medication
            med = patient[2][idx] + [-1] * (max_med_len - len(patient[2][idx]))
            med_list.append(med)
            m_mask_list.append([1] * len(patient[2][idx]) + [0] * (max_med_len - len(patient[2][idx])))

            target_list.append(patient[2][idx])

            if idx > 0:
                # previous visit info
                dec_disease = patient[0][idx - 1] + [-1] * (max_diag_len - len(patient[0][idx - 1]))
                stay_disease = patient[0][idx] + [-1] * (max_diag_len - len(patient[0][idx]))
                dec_disease_list.append(dec_disease)
                stay_disease_list.append(stay_disease)
                dec_disease_mask_list.append(
                    [1] * len(patient[0][idx - 1]) + [0] * (max_diag_len - len(patient[0][idx - 1])))

                dec_proc = patient[1][idx - 1] + [-1] * (max_proc_len - len(patient[1][idx - 1]))
                stay_proc = patient[1][idx] + [-1] * (max_proc_len - len(patient[1][idx]))
                dec_proc_list.append(dec_proc)
                stay_proc_list.append(stay_proc)
                dec_proc_mask_list.append(
                    [1] * len(patient[1][idx - 1]) + [0] * (max_proc_len - len(patient[1][idx - 1])))

        # padding for visits
        disease_list.extend([[0] * max_diag_len] * (max_visit - len(disease_list)))
        proc_list.extend([[0] * max_proc_len] * (max_visit - len(proc_list)))
        med_list.extend([[0] * max_med_len] * (max_visit - len(med_list)))
        d_mask_list.extend([[0] * max_diag_len] * (max_visit - len(d_mask_list)))
        p_mask_list.extend([[0] * max_proc_len] * (max_visit - len(p_mask_list)))
        m_mask_list.extend([[0] * max_med_len] * (max_visit - len(m_mask_list)))

        # padding for length lists
        d_length_list.extend([0] * (max_visit - len(d_length_list)))
        p_length_list.extend([0] * (max_visit - len(p_length_list)))
        m_length_list.extend([0] * (max_visit - len(m_length_list)))

        batch_disease.append(disease_list)
        batch_proc.append(proc_list)
        batch_med.append(med_list)
        batch_mask_d.append(d_mask_list)
        batch_mask_p.append(p_mask_list)
        batch_mask_m.append(m_mask_list)
        batch_d_length.append(d_length_list)
        batch_p_length.append(p_length_list)
        batch_m_length.append(m_length_list)
        batch_target.append(target_list)

        if len(dec_disease_list) > 0:
            dec_disease_list.extend([[0] * max_diag_len] * (max_visit - len(dec_disease_list) - 1))
            stay_disease_list.extend([[0] * max_diag_len] * (max_visit - len(stay_disease_list) - 1))
            dec_disease_mask_list.extend([[0] * max_diag_len] * (max_visit - len(dec_disease_mask_list) - 1))
            stay_disease_mask_list.extend([[0] * max_diag_len] * (max_visit - len(stay_disease_mask_list) - 1))

            dec_proc_list.extend([[0] * max_proc_len] * (max_visit - len(dec_proc_list) - 1))
            stay_proc_list.extend([[0] * max_proc_len] * (max_visit - len(stay_proc_list) - 1))
            dec_proc_mask_list.extend([[0] * max_proc_len] * (max_visit - len(dec_proc_mask_list) - 1))
            stay_proc_mask_list.extend([[0] * max_proc_len] * (max_visit - len(stay_proc_mask_list) - 1))

            batch_dec_disease.append(dec_disease_list)
            batch_stay_disease.append(stay_disease_list)
            batch_dec_disease_mask.append(dec_disease_mask_list)
            batch_stay_disease_mask.append(stay_disease_mask_list)

            batch_dec_proc.append(dec_proc_list)
            batch_stay_proc.append(stay_proc_list)
            batch_dec_proc_mask.append(dec_proc_mask_list)
            batch_stay_proc_mask.append(stay_proc_mask_list)

    return (torch.tensor(batch_disease),
            torch.tensor(batch_proc),
            torch.tensor(batch_med),
            torch.tensor([len(p[0]) for p in batch]),
            torch.tensor(batch_d_length),
            torch.tensor(batch_p_length),
            torch.tensor(batch_m_length),
            torch.tensor(batch_mask_d),
            torch.tensor(batch_mask_p),
            torch.tensor(batch_mask_m),
            torch.tensor(batch_dec_disease) if len(batch_dec_disease) > 0 else None,
            torch.tensor(batch_stay_disease) if len(batch_stay_disease) > 0 else None,
            torch.tensor(batch_dec_disease_mask) if len(batch_dec_disease_mask) > 0 else None,
            torch.tensor(batch_stay_disease_mask) if len(batch_stay_disease_mask) > 0 else None,
            torch.tensor(batch_dec_proc) if len(batch_dec_proc) > 0 else None,
            torch.tensor(batch_stay_proc) if len(batch_stay_proc) > 0 else None,
            torch.tensor(batch_dec_proc_mask) if len(batch_dec_proc_mask) > 0 else None,
            torch.tensor(batch_stay_proc_mask) if len(batch_stay_proc_mask) > 0 else None,
            batch_target)



def pad_batch_v2_eval(batch):
    """
    对评估数据进行padding和转换
    """
    return pad_batch_v2_train(batch)


def pad_num_replace(tensor, from_num, to_num):
    """
    替换tensor中的特定值
    """
    return torch.where(tensor == from_num, torch.tensor(to_num), tensor)