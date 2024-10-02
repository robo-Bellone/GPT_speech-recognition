'''
import whisper
import sounddevice as sd
import numpy as np

# 임계값 설정 (예: 0.5, 임의로 설정 가능)
threshold = 0.005

# Whisper 모델 로드
model = whisper.load_model("small")

# 콜백 함수: 오디오 스트림에서 발생하는 데이터를 처리하고 변환
def callback(indata, frames, time, status):
    if status:

        print(status)
        print("....")

    if np.any(indata):
    
        print("----")
        # NumPy 배열로 데이터 변환
        audio_data = np.frombuffer(indata, dtype=np.float32)  # 부동 소수점으로 변환

        # 임계값 이상인 샘플만 처리
        above_threshold = audio_data[audio_data > threshold]

        # 처리된 데이터가 임계값 이상인 경우에만 인식
        if np.any(above_threshold):
            # 오디오 데이터를 Whisper 모델에 전달하여 텍스트로 변환
            result = model.transcribe(above_threshold, language='ko')
            # 변환된 텍스트 출력
            print(result["text"])

# 오디오 스트림 시작 (마이크 입력)
with sd.InputStream(callback=callback):
    sd.sleep(500)
    '''
'''

whisper 사용시 기본 방법

'''

import whisper
import sounddevice as sd
import numpy as np

# Whisper 모델 로드
model = whisper.load_model("small")

# 콜백 함수: 오디오 스트림에서 발생하는 데이터를 처리하고 변환
def callback(indata, frames, time, status):
    if status:
        print(status)
    if np.any(indata):
        # NumPy 배열로 데이터 변환
        audio_data = np.frombuffer(indata, dtype=np.float32)  # 부동 소수점으로 변환
        # 오디오 데이터를 Whisper 모델에 전달하여 텍스트로 변환
        result = model.transcribe(audio_data, language='ko')
        # 변환된 텍스트 출력
        print(result["text"])

# 오디오 스트림 시작 (마이크 입력)
with sd.InputStream(callback=callback):
    sd.sleep(1000)

