from facenet_pytorch import MTCNN
from torchvision.models import vgg16
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

mtcnn = MTCNN(keep_all=True, post_process=False)
vggmodel = vgg16(pretrained='vggface2').eval()
# feature_vectors = np.load('./SaveFace/FaceFeature.npy')
feature_vectors = np.load('WOA7018-main/src/WOA7018/SaveFace/FaceFeature.npy')


def create_folder(folder_path):
    if os.path.exists(folder_path):
        return
    try:
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' create sucessfully")
    except OSError as e:
        print(f"create folder failed: {e}")

def cosine_similarity(matrix, vector):
    # 计算矩阵的每一行与向量的余弦相似度
    dot_product = np.dot(matrix, vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    # 避免除零错误
    cosine_similarities = dot_product / (matrix_norms * vector_norm + 1e-8)

    return cosine_similarities


def process_images(images, state=0, name=None):
    global feature_vectors

    flag = 0

    if name != None and state == 1:
        label = name

    each_label = []

    # 遍历所有图片
    for idx, img in enumerate(images):
        # 获取人脸标注框, 可能存在多张人脸
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            # 一张图片多张脸只保存第一张
            face = Image.fromarray(img).crop(boxes[0])
            # plt.imshow(face)
            face_tensor = mtcnn(face)
            # 转化为人脸特征向量
            feature_vector = vggmodel(face_tensor).detach().numpy()

            flag = 1

            # 状态为1时,生成人名文件夹,保存人脸图片
            if state == 1 and name != None:
                folder_name = f'WOA7018-main/src/WOA7018/SaveFace/{name}'
                create_folder(folder_name)
                # 将裁剪部分人脸保存在文件中
                face.save(f'{folder_name}/{idx}.jpg')
                # 特征向量与人名拼接
                featureVector = np.hstack((feature_vector[0], np.array([label])))
                #存储特征向量
                if len(feature_vectors) != 0:
                    feature_vectors = np.vstack((feature_vectors, featureVector))
                else:
                    feature_vectors = featureVector


            if state == 0:
                # 计算余弦相似度
                similarities = cosine_similarity(
                    feature_vectors[:,:-1].astype(np.float64), feature_vector[0])
                print(similarities)
                if np.max(similarities) > 0.93:
                    index = np.argmax(similarities)
                    prob = np.max(similarities)
                    each_label.append((feature_vectors[index, -1], prob))
                    return {'result': 0, 'name':feature_vectors[index, -1]}
                else:
                    continue
        # 该图片未检测到人脸 下一张
        else:
            continue

    if state == 0:
        # 所有图片未检测到人脸
        if flag == 0:
            return {'result': 1, 'name':'', "message":"No Face Deteect in Each Image"}
        # 检测到人脸 但与数据库不匹配
        if len(each_label) == 0:
            return {'result': 1, 'name':'', "message":"Invalid Acessce"}

    # state = 1, 进行人脸录入 保存向量
    np.save('WOA7018-main/src/WOA7018/SaveFace/FaceFeature.npy', feature_vectors)
    return {'result': 0, 'name':name}


