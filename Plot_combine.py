import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 이미지가 저장된 디렉터리 경로
base_dir ="G:/My Drive/2. 연구/3. hand gesture recognition using IMU data/log_seq_250311/w_o object/"
exp = "250310_1_hr"
# exp = "250310_1_prof"


directory = base_dir+exp+'/figure_raw_data/'

# 파일명에서 [Label #, Segment #] 패턴 찾기
pattern = re.compile(r'\[Label (\d+), Segment (\d+)\] cut20')
labels_segments = set()
image_files = []

for filename in os.listdir(directory):
    match = pattern.search(filename)
    if match:
        labels_segments.add(match.group())
        image_files.append(os.path.join(directory, filename))

# Label, Segment 조합 개수 확인
num_combinations = len(labels_segments)
print(num_combinations)
image_dict = {}
for img_file in image_files:
    match = pattern.search(img_file)
    if match:
        label, segment = match.groups()
        key = (int(label), int(segment))
        if key not in image_dict:
            image_dict[key] = {'Acceleration': None, 'Gyroscope': None, 'Joint Position': None, 'Joint Angles': None}
        
        # 파일 종류에 따라 저장
        if 'Acceleration' in img_file:
            image_dict[key]['Acceleration'] = img_file
        elif 'Gyroscope' in img_file:
            image_dict[key]['Gyroscope'] = img_file
        elif 'Joint Position' in img_file:
            image_dict[key]['Joint Position'] = img_file
        elif 'Joint Angles' in img_file:
            image_dict[key]['Joint Angles'] = img_file
# 각 서브플롯에 이미지 추가
for label_segment in labels_segments:
    match = pattern.search(label_segment)
    if match:
        label, segment = match.groups()
        key = (int(label), int(segment))  # image_dict의 키 형식으로 변환
        if key in image_dict:  # 이제 올바른 키로 검색
            img_files = [
                image_dict[key]['Acceleration'],
                image_dict[key]['Gyroscope'],
                image_dict[key]['Joint Position'],
                image_dict[key]['Joint Angles']
            ]            
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            fig.suptitle(f"{exp}-{label_segment}", fontsize=10, y=0.98)
            
            for ax, img_path in zip(axes.flat, img_files):
                if img_path:  # 이미지 파일이 존재하는 경우에만
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis("off")  # 축 제거
            plt.savefig(f'{base_dir}/combined_figure/{exp}_{label_segment}_frame20.png', 
                        bbox_inches='tight', 
                        dpi=300)
            plt.close()
# # 각 조합에 대해 figure 생성
# for i, label_segment in enumerate(labels_segments):
#     # Figure 생성 - 더 큰 크기로 설정
    
#     fig, axes = plt.subplots(2, 2, figsize=(20, 10))
#     fig.suptitle(f"{exp}-{label_segment}", fontsize=10, y=0.98)

#     # 각 서브플롯에 이미지 추가
#     for ax, img_path in zip(axes.flat, image_files[i*4:(i+1)*4]):
#         img = mpimg.imread(img_path)
#         ax.imshow(img)
#         ax.axis("off")  # 축 제거
    
#     # 서브플롯 간격 조절
#     plt.subplots_adjust(
#         left=0.05,    # 왼쪽 여백
#         right=0.95,   # 오른쪽 여백
#         bottom=0.05,  # 아래쪽 여백
#         top=0.95,     # 위쪽 여백
#         wspace=0.2,   # 가로 방향 서브플롯 간격
#         hspace=0.2    # 세로 방향 서브플롯 간격
#     )
    
#     # Figure 저장 및 출력
#     plt.savefig(f'{base_dir}/combined_figure/{exp}_{label_segment}_frame60.png', 
#                 bbox_inches='tight', 
#                 dpi=300)
#     # plt.show()
#     plt.close()  # 메모리 관리를 위해 figure 닫기
