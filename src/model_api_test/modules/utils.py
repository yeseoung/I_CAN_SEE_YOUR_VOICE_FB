import numpy as np

# vector normalization (학습 프로젝트 sing_lang_trans/modules/utils.py와 같은 계산)
def Vector_Normalization(joint):
    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
    v = v2 - v1
    # Normalize v safely (겹친 관절은 길이 0)
    norm = np.linalg.norm(v, axis=1)[:, np.newaxis]
    norm = np.where(norm == 0, 1e-6, norm)
    v = v / norm

    # 일직선 관절에서 float 오차로 내적이 1을 살짝 넘으면 arccos가 NaN이 되므로 자른다
    dot = np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    v = np.nan_to_num(v, nan=0.0)
    angle_label = np.nan_to_num(angle_label, nan=0.0)

    return v, angle_label


# 지문자 v2의 추가 특징 (학습 프로젝트 modules/features.py letter_features의 "lengths" 묶음과 같은 계산)
def Bone_Length_Ratio(joint):
    """뼈 20개 길이 / 손바닥 길이(손목~가운데 손가락 뿌리). 손바닥 길이가 0에 가까우면 0."""
    bone = np.linalg.norm(
        joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
        - joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]], axis=1)
    palm = np.linalg.norm(joint[9] - joint[0])
    if palm <= 1e-6:
        return np.zeros(20, dtype=np.float32)
    return (bone / palm).astype(np.float32)
