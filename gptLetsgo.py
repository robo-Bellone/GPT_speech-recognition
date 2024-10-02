import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write
import time
import subprocess as sbp
import os
from pydub import AudioSegment

status_file = "status_file.txt"

assistance_file = "STmemory_assistance_file.txt"
user_file = "STmemory_user_file.txt"


def send_termination_signal():
    with open(status_file, 'w') as file:
        file.write("terminate")

# 매개변수 설정
SAMPLE_RATE = 44100  # 샘플링 레이트
DURATION = 0.1  # 녹음 지속 시간 (초)
FRAMES = int(SAMPLE_RATE * DURATION)  # 프레임 수
SMOOTHING_SIZE = 5  # 스무딩 윈도우 크기
THRESHOLD = 40  # 발화 감지 임계값
SILENCE_DURATION = 0.5  # 녹음 중지를 위한 침묵 지속 시간 (초)

start_time = time.time()

# 오디오 및 발화 상태 데이터
audio_data = None
recording = False
last_speech_time = time.time()
recorded_audio = []

start_time = time.time()
# 1초 버퍼 초기화
buffer_duration = 0.8  # 버퍼 지속 시간 (초)
buffer_size = int(SAMPLE_RATE * buffer_duration)  # 버퍼 크기
audio_buffer = np.zeros((buffer_size, 1), dtype=np.float32)

#n초 입력 

# 콜백 함수 정의
def sound_callback(indata, frames, time_set, status):
    global audio_data, recording, last_speech_time, recorded_audio, audio_buffer, start_time
    
    if time.time() - start_time < 3:
        return 
    audio_data = indata[:, 0]

    if not recording:
        # 녹음이 시작되지 않았다면 버퍼를 업데이트합니다.
        audio_buffer = np.roll(audio_buffer, -len(indata), axis=0)
        audio_buffer[-len(indata):] = indata
    elif recording and len(recorded_audio) == 0:
        # 녹음이 시작된 직후, 버퍼링을 멈추고 녹음 데이터에 버퍼를 추가합니다.
        recorded_audio.append(audio_buffer.copy())

    if recording:
        recorded_audio.append(indata.copy())

# 오디오 스트림 초기화
stream = sd.InputStream(callback=sound_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAMES)
stream.start()

# Matplotlib 차트 초기화
fig, ax = plt.subplots()
x = np.arange(0, SAMPLE_RATE // 2, SAMPLE_RATE / FRAMES)
line, = ax.plot(x, np.zeros_like(x))
ax.set_xlim(0, SAMPLE_RATE // 2)
ax.set_ylim(0, 50)  # Y축 범위 조정 필요

# 발화 상태 표시 텍스트
status_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

# 발화 감지 카운터
speech_detected_count = 0

def update_plot(frame):
    global audio_data, recording, last_speech_time, recorded_audio, speech_detected_count, audio_buffer
    if audio_data is not None:
        # FFT 수행
        fft_result = np.fft.fft(audio_data)
        freq = np.fft.fftfreq(len(fft_result), 1 / SAMPLE_RATE)
        fft_abs = np.abs(fft_result)

        # 스무딩 적용
        smoothed_fft = np.convolve(fft_abs, np.ones(SMOOTHING_SIZE)/SMOOTHING_SIZE, mode='same')

        # FFT 결과 업데이트
        line.set_ydata(smoothed_fft[:FRAMES // 2])

        # 발화 상태 판별
        mean_amplitude = np.abs(fft_abs[0:1500]).max()
        if mean_amplitude > THRESHOLD:
            last_speech_time = time.time()
            if not recording:
                speech_detected_count += 1
                if speech_detected_count >= 2:
                    recording = True
                    send_termination_signal()
                    recorded_audio = []
            status_text.set_text(f"Speaking, audio level is {mean_amplitude}")
        else:
            if recording and (time.time() - last_speech_time) > SILENCE_DURATION:
                recording = False
                speech_detected_count = 0
                # 녹음된 오디오 파일 저장
                audio_to_save = np.concatenate(recorded_audio, axis=0)
                write('recorded_audio.wav', SAMPLE_RATE, audio_to_save)
                args = ['python3', 'new_stt.py']
                sbp.Popen(args)
                # 녹음 데이터 초기화
                recorded_audio = []

                # MP3 파일 로드 및 처리
                dukup_audio = AudioSegment.from_file("/home/jongil/whisper/dukup.mp3", format="mp3")
                dukup_samples = np.array(dukup_audio.get_array_of_samples(), dtype=np.float32) / 50000
                fs = int(dukup_audio.frame_rate * 0.9)
                dukup_samples = dukup_samples[:(dukup_samples.size // 3) * 3].reshape(-1, 3).mean(axis=1)

                sd.play(dukup_samples, fs)

                #print('started')
            status_text.set_text(f"Silent")

    return line, status_text

# 애니메이션 실행
ani = FuncAnimation(fig, update_plot, blit=True, save_count=100)
plt.show()


if os.path.exists(assistance_file):
    os.remove(assistance_file)

if os.path.exists(user_file):
    os.remove(user_file)