from gtts import gTTS
import sounddevice as sd
import argparse
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import os
import time

status_file = "status_file.txt"


def send_termination_signal():
    with open(status_file, 'w') as file:
        file.write("terminate")

    
def check_termination_signal():
    if os.path.exists(status_file):
        with open(status_file, 'r') as file:
            content = file.read().strip()
            if content == "terminate":
                return True
    return False
    
def clear_termination_signal():
    if os.path.exists(status_file):
        os.remove(status_file)


def main():
    parser = argparse.ArgumentParser(description='띄어쓰기가 포함된 텍스트 인자를 처리하는 예제')
    parser.add_argument('text', type=str, nargs='+', help='입력할 텍스트 (띄어쓰기 포함 가능)')

    args = parser.parse_args()
    combined_text = ' '.join(args.text)
    print(f'입력된 텍스트: {combined_text}')

    sd.stop()
    tts_ko = gTTS(text=combined_text, lang='ko')
    mp3_fp = BytesIO()
    tts_ko.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 50000
    fs = int(audio.frame_rate * 1.7)
    #/home/jongil/whisper/dukup.mp3


    ############################
        # 추가할 MP3 파일 로드
    additional_audio = AudioSegment.from_file("/home/jongil/whisper/dukup.mp3", format="mp3")
    additional_samples = np.array(additional_audio.get_array_of_samples(), dtype=np.float32) / 50000

    additional_samples = additional_samples[:(additional_samples.size // 3) * 3].reshape(-1, 3).mean(axis=1)

    # 배열 합치기
    combined_samples = np.concatenate((samples, additional_samples))

    # 오디오 재생 시작
    sd.play(combined_samples, fs)
    #sd.play(samples, fs)

    # 종료 신호 확인
    while sd.get_stream().active:
        if check_termination_signal():
            print("종료 신호가 감지되었습니다. 프로세스를 종료합니다.")
            sd.stop()
            clear_termination_signal()
            break
        time.sleep(0.1)

if __name__ == '__main__':
    clear_termination_signal()
    main()
