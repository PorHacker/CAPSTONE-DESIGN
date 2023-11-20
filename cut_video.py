import os
import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# JSON 파일에서 필요한 정보를 읽어옵니다.
def read_metadata_and_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:  # utf-8 인코딩으로 지정
        data = json.load(f)
    video_name = data['metaData']['name']
    video_data = data['data']
    return video_name, video_data

# 주어진 영상 파일을 자르고 저장합니다.
def extract_and_save_video(input_video, output_video, start_time, end_time):
    ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)

# JSON 파일 경로와 영상 디렉토리 설정
json_file = "D:/Pycharm/gesture-recognition-master_V.2.0/dataset/video/morpheme/NIA_SL_SEN0205_REAL01_D_morpheme.json"  # JSON 파일의 경로
video_directory = "D:/Pycharm/gesture-recognition-master_V.2.0/dataset/video/pre_video"  # 영상 파일이 저장된 디렉토리 경로
output_directory = "D:/Pycharm/gesture-recognition-master_V.2.0/dataset/video/cut_video"  # 추출된 영상을 저장할 디렉토리 경로

# JSON 파일에서 필요한 정보를 읽어옵니다.
video_name, video_data = read_metadata_and_data(json_file)

# 영상 파일 경로 설정
input_video_path = os.path.join(video_directory, video_name)

# "data" 내의 각 항목에 대해 영상을 자르고 저장합니다.
for item in video_data:
    start_time = item['start']
    end_time = item['end']
    attribute_name = item['attributes'][0]['name']
    output_video_name = f"{video_name.replace('.mp4','')}_{attribute_name}.mp4"
    output_video_path = os.path.join(output_directory, output_video_name)

    # 최소 길이를 0.5로 유지하기 위한 조정
    if end_time - start_time < 0.5:
        # 시간을 추가하여 최소 길이를 0.5로 유지
        adjust = (0.5 - (end_time - start_time)) / 2
        start_time -= adjust
        end_time += adjust
        
    extract_and_save_video(input_video_path, output_video_path, start_time, end_time)

    print(f'영상 {output_video_name}을 저장했습니다.')
