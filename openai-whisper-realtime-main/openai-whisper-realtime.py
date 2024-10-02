
'''
import sounddevice as sd
import numpy as np

import whisper

import asyncio
import queue
import sys


# SETTINGS
MODEL_TYPE="small"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="Korean"
# pre-set the language to avoid autodetection
#BLOCKSIZE=24678 
BLOCKSIZE=24678
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds. 

#SILENCE_THRESHOLD=400
SILENCE_THRESHOLD=1000
# should be set to the lowest sample amplitude that the speech in the audio material has
#SILENCE_RATIO=100
SILENCE_RATIO=50
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
model = whisper.load_model("small")

async def inputstream_generator():
	"""Generator that yields blocks of input data as NumPy arrays."""
	q_in = asyncio.Queue()
	loop = asyncio.get_event_loop()

	def callback(indata, frame_count, time_info, status):
		loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

	stream = sd.InputStream(samplerate=22000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
	with stream:
		while True:
			indata, status = await q_in.get()
			#print('----')
			yield indata, status


############################################
async def process_audio_buffer():
	global global_ndarray
	async for indata, status in inputstream_generator():

		indata_flattened = abs(indata.flatten())

		# discard buffers that contain mostly silence
		if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO) and False:
			continue

		if (global_ndarray is not None):
			global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
		else:
			global_ndarray = indata

		# concatenate buffers if the end of the current buffer is not silent
		if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
			#print('tlqkf')
			continue
		else:
			local_ndarray = global_ndarray.copy()
			global_ndarray = None
			indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
			result = model.transcribe(indata_transformed, language=LANGUAGE)
			print(result["text"])

		del local_ndarray
		del indata_flattened


async def main():
	print('\nActivating wire ...\n')
	audio_task = asyncio.create_task(process_audio_buffer())
	while True:
		await asyncio.sleep(0.5)
	audio_task.cancel()
	try:
		await audio_task
	except asyncio.CancelledError:
		print('\nwire was cancelled')


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')
'''

'''
import sounddevice as sd
import numpy as np
import whisper
import asyncio
import queue
import sys


# 설정
MODEL_TYPE = "small"  # 변환에 사용할 모델 종류
LANGUAGE = "Korean"  # 사전 설정된 언어로 자동 감지를 피함
BLOCKSIZE = 24678  # 오디오를 나누는 기본 청크 크기(샘플 단위)
SILENCE_THRESHOLD = 500  # 오디오 재료에서 음성의 최소 샘플 진폭 설정
SILENCE_RATIO = 50  # 버퍼 내 임계값보다 높을 수 있는 샘플 수

global_ndarray = None  # 전역 넘파이 배열 초기화
model = whisper.load_model("small")  # Whisper 모델 로드

async def inputstream_generator():
    """NumPy 배열로 오디오 데이터 블록을 생성하는 생성기."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        # 콜백 함수, 오디오 데이터를 큐에 넣음
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    # 오디오 입력 스트림 설정
    stream = sd.InputStream(samplerate=22000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def process_audio_buffer():
    global global_ndarray
    async for indata, status in inputstream_generator():
        # 입력 데이터 평탄화
        indata_flattened = abs(indata.flatten())

        # 주로 침묵이 포함된 버퍼를 무시
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO) and False:
            continue

        # 전역 배열에 오디오 데이터 추가
        if (global_ndarray is not None):
            global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
        else:
            global_ndarray = indata

        # 버퍼의 끝 부분이 침묵이 아니면 계속 데이터 추가
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
            continue
        else:
            # 현재 버퍼의 복사본을 만듦
            local_ndarray = global_ndarray.copy()
            global_ndarray = None
            # 오디오 데이터를 Whisper 모델이 인식할 수 있는 형태로 변환
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            # Whisper 모델을 사용하여 텍스트 변환
            result = model.transcribe(indata_transformed, language=LANGUAGE)

            # 'MBC 뉴스'가 포함되어 있지 않으면 결과 출력 (추가된 부분)
            if 'MBC 뉴스' not in result["text"]:
                print(result["text"])

        # 사용된 배열 삭제
        del local_ndarray
        del indata_flattened

# 메인 함수
async def main():
    print('\nActivating wire ...\n')
    # 오디오 처리 작업 생성
    audio_task = asyncio.create_task(process_audio_buffer())
    while True:
        await asyncio.sleep(0.5)
    # 작업 취소
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')

# 스크립트 실행 부분
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
'''


import sounddevice as sd
import numpy as np
import torch

import whisper

import asyncio
import queue
import sys


# SETTINGS
MODEL_TYPE="small"  # Whisper 모델의 종류 설정 ("small" 모델 사용). Whisper는 다양한 언어를 지원하는 음성 인식 모델입니다.
LANGUAGE="Korean"  # 음성 인식 언어 설정 (한국어로 설정). 이 언어는 Whisper 모델에 의해 인식될 언어를 지정합니다.
BLOCKSIZE=4000  # 오디오를 나누는 청크 크기 설정 (샘플 단위). 이 크기는 오디오가 처리될 때 나누어지는 덩어리의 크기를 결정합니다.
SILENCE_THRESHOLD_1 = 200  # 음성 인식에 사용할 최소 샘플 진폭 설정. 이 값보다 낮은 진폭의 소리는 침묵으로 간주됩니다.
# 이건 말 인식할때
SILENCE_THRESHOLD_2 = 50
# 이건 말 출력
SILENCE_RATIO=50  # 버퍼 내에서 임계값을 초과할 수 있는 샘플 수 설정. 이 비율은 얼마나 많은 소리가 침묵인지를 판단하는 기준이 됩니다.

global_ndarray = None  # 글로벌 오디오 배열을 초기화합니다. 이 배열은 오디오 데이터를 임시 저장하는 데 사용됩니다.
model = whisper.load_model(MODEL_TYPE)  # Whisper 모델을 로드합니다. 이 모델은 나중에 오디오 데이터를 텍스트로 변환하는 데 사용됩니다.

async def inputstream_generator():## 오디오 받는 곳
    """오디오 데이터를 NumPy 배열로 제공하는 생성기(generator). 이 함수는 오디오 스트림에서 데이터를 읽고 그것을 처리할 준비를 합니다."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        # 오디오 스트림에서 데이터를 받아 큐에 넣는 콜백 함수. 이 함수는 오디오 데이터를 비동기적으로 처리할 수 있게 합니다.
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback)
    # 오디오 입력 스트림을 설정합니다. 여기서 샘플레이트, 채널 수, 데이터 타입, 블록 크기 등을 설정합니다.
    with stream:
        while True:
            indata, status = await q_in.get()
            # 큐에서 오디오 데이터 블록을 받아옵니다. 이 데이터는 나중에 Whisper 모델로 전송됩니다.
            yield indata, status



async def process_audio_buffer():
    global global_ndarray
    async for indata, status in inputstream_generator():


        indata_flattened = abs(indata.flatten())
        # 오디오 데이터를 평탄화하고 절대값을 취합니다. 이는 데이터를 처리하기 쉽게 만듭니다.

        # 대부분 침묵인 버퍼는 무시합니다. 이는 처리해야 할 데이터의 양을 줄이고, 노이즈를 줄이는 데 도움이 됩니다.
		# 스레스홀드보다 작으면 무시하고 계속 돔
        if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD_1)).size < SILENCE_RATIO) and False:

            print("아")
            
            continue

        # 오디오 데이터 배열에 현재 버퍼를 추가합니다. 이 배열은 나중에 Whisper 모델에 전달됩니다.
        if (global_ndarray is not None):
            global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
            print("나")
        else:
            global_ndarray = indata
            print("시")


###############################################################################################################
        # 현재 버퍼 끝이 침묵이 아니면 계속 추가합니다. 이는 말이 끊기지 않도록 연속된 음성 데이터를 유지하기 위함입니다.
		# 스레스 홀드가 작으면 루프를 계속돌 가능성 있음
        if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD_2):
            print("말 하는중")
            continue
        
		## 버퍼 끝을 봤는데 말이 없으면 프린트
        else:
            print("말 끝")
            local_ndarray = global_ndarray.copy()
            global_ndarray = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            # 오디오 데이터를 Whisper 모델이 처리할 수 있는 형태로 변환합니다.
            indata_transformed_tensor = torch.tensor(indata_transformed)
            result = model.transcribe(indata_transformed_tensor, language=LANGUAGE)
            # Whisper 모델을 사용하여 오디오 데이터를 텍스트로 변환합니다.
            if result["text"] != "":
                print("-------------------------------------------")
                print(result["text"])
                print("-------------------------------------------")
            # 변환된 텍스트가 있다면 출력합니다.

        del local_ndarray
        del indata_flattened

async def main():
    print('\nActivating wire ...\n')
    # 메인 함수를 시작합니다. 여기서는 오디오 처리 작업을 시작합니다.
    audio_task = asyncio.create_task(process_audio_buffer())
    # 오디오 버퍼 처리 작업을 비동기적으로 시작합니다.
    while True:
        await asyncio.sleep(1)
    # 프로그램이 계속 실행되도록 유지합니다.
    audio_task.cancel()
    # 오디오 처리 작업을 취소합니다.
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')
    # 오디오 처리 작업이 취소되었을 때 처리합니다.

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
    # 프로그램이 사용자에 의해 중단되었을 때 처리합니다.
