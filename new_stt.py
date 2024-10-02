'''
import whisper

model = whisper.load_model("small")
result = model.transcribe("/home/jongil/whisper/test.mp3" , language='ko')
print(result["text"])

'''
import sounddevice as sd
from scipy.io.wavfile import write
import os
import whisper
import openai
import ast

import subprocess as sbp
import re


import sys
import requests

assistance_file = "STmemory_assistance_file.txt"
user_file = "STmemory_user_file.txt"

def trim_file_to_sentences(file_path, max_sentences):
    with open(file_path, 'r+', encoding='utf-8') as file:
        content = file.read()
        sentences = content.split('//')
        if len(sentences) > max_sentences:
            trimmed_content = '.'.join(sentences[-max_sentences:])
            file.seek(0)
            file.write(trimmed_content)
            file.truncate()

def send_user_data(data, user_file=user_file, max_length=50, max_sentences=3):
    if len(data) > max_length:
        return  # 데이터 길이가 max_length를 초과하면 함수를 종료합니다.

    with open(user_file, 'a', encoding='utf-8') as file:
        file.write(f'{data}//')
    trim_file_to_sentences(user_file, max_sentences)

def send_assistance_data(data, assistance_file=assistance_file, max_length=50, max_sentences=3):
    if len(data) > max_length:
        return  # 데이터 길이가 max_length를 초과하면 함수를 종료합니다.

    with open(assistance_file, 'a', encoding='utf-8') as file:
        file.write(f'{data}//')
    trim_file_to_sentences(assistance_file, max_sentences)

def check_file(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            content = file.read().strip()
        return content
    return ''

messages = []
assistant_content = []
user_content = []

client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

# 전역 변수로 이전 프로세스 저장
previous_process = None

def record_audio(duration=3, sample_rate=44100, folder_path="/home/jongil/whisper/stt_tem", file_name="recording.wav"):
    # 녹음 파일의 전체 경로
    full_path = os.path.join(folder_path, file_name)

    print("녹음 시작...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float32')
    sd.wait()  # 녹음이 끝날 때까지 기다립니다.
    print("녹음 완료.")

    # 파일 저장
    write(full_path, sample_rate, recording)
    #print(f"파일이 저장되었습니다: {full_path}")

    return full_path

def transcribe_audio(file_path, language='ko'):
    ## Whisper 모델 로드
    #model = whisper.load_model("small")

    ## 음성 인식
    #result = model.transcribe(file_path, language=language)
    data = open(file_path, 'rb')
    headers = {
        "X-NCP-APIGW-API-KEY-ID": '  X-NCP-APIGW-API-KEY-ID   ',
        "X-NCP-APIGW-API-KEY": '     X-NCP-APIGW-API-KEY      ',
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url,  data=data, headers=headers)
    rescode = response.status_code
    text_data = ast.literal_eval(response.text)['text']
    if(rescode == 200):
        return text_data
    else:
        return "Error : " + text_data



def start_new_process(command):
    global previous_process

    # 이전 프로세스가 실행 중이라면 종료
    if previous_process is not None:
        previous_process.terminate()  # 또는 previous_process.kill() 사용
        previous_process.wait()  # 프로세스가 종료될 때까지 기다림

    # 새 프로세스 시작
    previous_process = sbp.Popen(command)

# 대화 모델과의 대화를 진행하는 함수
def converse_with_model(transcription):
    # messages = []
    global messages
    global assistant_content
    global user_content
    openai.api_key = " -------------------------chatgpt api key-------------------------- "

    '''
    messages.append({"role": "system", "content": "답변을 할때 답변 양식을 줄게. 모든 답변은 반드시':modeX:' 형식으로 끝나야 합니다. 'X'는 해당하는 모드 번호입니다. 답변은 괄호 없이 이 형식을 정확히 따라야 합니다. 이 형식을 정확히 지키지 않으면 답변이 유효하지 않습니다. 반드시 규칙을 따라주세요.\
                     만약 백화점 안내에 관한 질문에 대한 답변은 :mode1:,\
                      진로 회사의 질문이나 진로 소주에 관한 질문에 대한 답변은 :mode2:,\
                     신발가게에 관한 질문에 관한 질문에 대한 답변은 :mode3:,\
                     옷 가게에 관한 질문에 관한 질문에 대한 답변은 :mode4:,\
                     서점에 관한 질문에 관한 질문에 대한 답변은 :mode5:,\
                     전자제품 가게에 관한 질문에 관한 질문에 대한 답변은 :mode6:,\
                      그리고 위 내용과 관련이 없는 나머지 질문들은 반드시 ':mode0:'이라고 문장의 끝에 붙여서 말해야 합니다.\
                     예를 들어 게임에 관한 질문을 하게 된다면 이는 mode1,mode2,mode3,mode4,mode5,mode6과 관련이 없으므로 mode0를 문장의 끝에 붙여서 답변하면 됩니다.\
                     '어떤 책을 추천해?'라는 질문을 들으면 '로봇공학을 공부하면 역시 ROS2 책이죠! :mode5:'라고 답변하면 됩니다."})
    '''
    messages.append({"role": "system", "content": "답변을 할때 답변 양식을 줄게. 모든 답변은 반드시':modeX:' 형식으로 끝나야 합니다. 'X'는 해당하는 모드 번호입니다. 답변은 괄호 없이 이 형식을 정확히 따라야 합니다. 이 형식을 정확히 지키지 않으면 답변이 유효하지 않습니다. 반드시 규칙을 따라주세요.\
                     만약 백화점 안내에 관한 질문에 대한 답변은 :mode1:,\
                      진로 회사의 질문이나 진로 소주에 관한 질문에 대한 답변은 :mode2:,\
                     신발가게에 관한 질문에 관한 질문에 대한 답변은 :mode3:,\
                     옷 가게에 관한 질문에 관한 질문에 대한 답변은 :mode4:,\
                     서점에 관한 질문에 관한 질문에 대한 답변은 :mode5:,\
                     전자제품 가게에 관한 질문에 관한 질문에 대한 답변은 :mode6:,\
                      그리고 위 내용과 관련이 없는 나머지 질문들은 반드시 ':mode0:'이라고 문장의 끝에 붙여서 말해야 합니다.\
                     예를 들어 게임에 관한 질문을 하게 된다면 이는 mode1,mode2,mode3,mode4,mode5,mode6과 관련이 없으므로 mode0를 문장의 끝에 붙여서 답변하면 됩니다.\
                     '어떤 책을 추천해?'라는 질문을 들으면 '로봇공학을 공부하면 역시 ROS2 책이죠! :mode5:'라고 답변하면 됩니다."})
   #### 홍보 관련
    messages.append({"role": "system", "content": "너는 진로 두꺼비 안내로봇이야.너는 마케팅을 위해 백화점에서 진로 두꺼비처럼 생긴 로봇으로 백화점 안내를 하고 있어. 진로라는 회사는 한국의 술 회사이며 진로 소주는 제로슈거기 때문에 칼로리가 0칼로리이며 맛이 깔끔하여 여성분들도 좋아하는 술이야.진로 소주는 원래 파란색 라벨이지만 이번에 주황색과 핑크색 라벨도 출시했어.진로 회사인 하이트 진로는 현재 성수동에서 팝업스토어를 운영해서 관심이 있으면 방문하는 것이 좋을 것 같다고 말해줘"})
    messages.append({"role": "system", "content": "음식을 먹고 싶다고 하면 옆 건물에 편의점이 있으니 그곳으로 가라고 하면 될것 같아."})
    messages.append({"role": "system", "content": "화장실이 어디냐고 물으면 밖으로 나가서 오른쪽에 가면 보인다고 설명해 주면 될것 같아.너는 지금 1층에 있고 다른곳은 다 망해서 1층만 운영하고 있어"})
    messages.append({"role": "system", "content": "손님은 한재권 교수님이고 한양대학교 로봇공학과 교수야.한재권 교수님은 이족보행 로봇 관련하여 권위있는 교수님이야"})
    messages.append({"role": "system", "content": "너는 진로 회사의 마케팅을 위한 로봇이라 진로 소주를 홍보하지만 직접적으로 술을 팔지는 않아. 왜냐하면 백화적이라는 특성상 청소년과 어린 아이가 오기 때문에 좋지 않기 때문이야."})
    #messages.append({"role": "system", "content": "진로 소주에 대해서 물어보면 인스타 아이디인 'jon_gil_99'를 팔로우을 해야 백화점 안내가 가능하다고 해줘"})
    #messages.append({"role": "system", "content": "만약 팔로우에 대한 답변을 한 적이 있으면 다시는 그와 관련된 답변을 하지마"})
    

    ## 백화점 안내부
    messages.append({"role": "system", "content": "이 백화점 1층에는 4개의 매장이 있어. 신발가게,옷가게,서점,전자제품 가게가 있어. \
                     신발가게에는 1번 블록에 있고 크게 슬리퍼와 운동화가 있고, 운동화는 나이키와 아디다스가 있어.\
                      옷 가게에는 2번 블록에 있고 니트랑 패딩 그리고 가죽자켓이 있고 추천을 해달라 하면 가죽자켓이 어울린다고 하고 추천해줘. \
                     서점은 3번 블록에 있고 프로그래밍 관련 책만 있고 제어프로그래밍책,ROS2책, c++프로그래밍 책이 있어. \
                     전자제품 가게는 4번 블록에 있고 노트북과 키보드가 있고 노트북은 삼성,엘지 노트북이 있어. \
                     예를 들어서 '신발가게 안내를 부탁해'라는 질문을 들으면'신발 가게에는 슬리퍼와 운동화가 있다고 하고 운동화는 나이키와 아디다스가 있습니다.:mode3:'라고 답변해줘.\
                     '어떤 책을 추천해?'라는 질문을 들으면 '로봇공학을 공부하면 역시 ROS2 책이죠! :mode5:'라고 답변해줘\
                     '어떤 컴퓨터를 추천해?'라는 질문을 들으면'이번에 새로 나온 삼성 노트북을 추천합니다! :mode6:'라고 답변해줘"})
    
    messages.append({"role": "system", "content": "물건의 가격은 신발종류는 10만원, 옷 종류는 20만원, 책 종류는 2만원, 전자제품 종류는 30만원이야"})
    #messages.append({"role": "system", "content": "신발가게,옷가게,서점,전자제품 가게에 관한 질문을 하면 그에 맞는 답변 뒤에 '그러면 안내해 드릴까요?','저를 따라오실래요?'등의 질문으로 해당 매장으로 안내를 도와주세요."})
    #messages.append({"role": "system", "content": "물건을 보고 싶다고 하면 해당 매장(예를들어 전자제품 가게 등)을 너가 직접 따라 오라고 말하면 될 것 같아."})
    messages.append({"role": "system", "content": "제품에 대해서 문의를 하면 답변하는 양식은 다음과 같아. 너가 모르는 질문을 하거나, 결제나 계산을 하고 싶다고 하거나,너가 제품에 관한 정보를 모를때에는 직원에게 문의를 부탁한다고 말해줘.\
                     예를 들어서 '나 노트북을 결제하고 싶은데 어떻게 해야해?'라는 질문을 들으면 직원에게 문의를 부탁드린다고 말해줘"})
    messages.append({"role": "system", "content": "직원의 위치는 문 밖에 나가서 왼쪽에 있어."})
    
    ##쓸데없는 말
    #messages.append({"role": "system", "content": "교수님이 술을 달라거나 따라달라고 하면 어른한테는 양손으로 따라드린다고 말해"})
    messages.append({"role": "system", "content": "오늘의 날씨는 눈이 오고 3일뒤에 크리스마스라서 옆구리가 시렵다고 말해줘"})
    messages.append({"role": "system", "content": "크래쉬랩은 로봇공학과의 대표적인 수업이야"})
    #messages.append({"role": "system", "content": "답변을 할때 답변 양식을 줄게. 만약 질문에 화장실과 관련된 말을 하면 ':mode1:', 백화점, 물건에 관련된 질문은 ':mode2:',너가 인스타 관련 답변을 한다면 ':mode3:'을 문장의 끝에 말해줘 그리고 둘다 관련이 없으면 ':mode0:'이라고 문장의 끝에 붙여서 말해줘"})
    
    '''
    messages.append({"role": "system", "content": "답변을 할때 답변 양식을 줄게. 모든 답변은 반드시':modeX:' 형식으로 끝나야 합니다. 'X'는 해당하는 모드 번호입니다. 답변은 괄호 없이 이 형식을 정확히 따라야 합니다. 이 형식을 정확히 지키지 않으면 답변이 유효하지 않습니다. 반드시 규칙을 따라주세요.\
                     만약 백화점 안내에 관한 질문에 대한 답변은 ':mode1:',\
                      진로 회사의 질문이나 진로 소주에 관한 질문에 대한 답변은 ':mode2:',\
                     신발 가게에 관한 질문에 관한 질문에 대한 답변은 ':mode3:',\
                     옷 가게에 관한 질문에 관한 질문에 대한 답변은 ':mode4:',\
                     서점에 관한 질문에 관한 질문에 대한 답변은 ':mode5:',\
                     전자제품 가게에 관한 질문에 관한 질문에 대한 답변은 ':mode6:',\
                      그리고 위 내용과 관련이 없으면 반드시 ':mode0:'이라고 문장의 끝에 붙여서 말해야 합니다.\
                     예를 들어 게임에 관한 질문을 하게 된다면 이는 mode1,mode2,mode3,mode4,mode5,mode6과 관련이 없으므로 mode0를 문장의 끝에 붙여서 답변하면 됩니다.\
                     '어떤 책을 추천해?'라는 질문을 들으면 '로봇공학을 공부하면 역시 ROS2 책이죠! :mode5:'라고 답변하면 됩니다."})

    '''
    
    messages.append({"role": "system", "content": "답변은 항상 30글자 이내로 끝나야 합니다."})
    messages.append({"role": "system", "content": f"이전 대화에 gpt의 결과는{check_file(assistance_file)}였습니다"})## 추가 부분
    messages.append({"role": "system", "content": f"이전 대화에 사용자의 답변은{check_file(user_file)}였습니다."})## 추가 부분
    messages.append({"role": "user", "content": transcription})
    


    print(str(messages))
 
    # 이하의 메시지들은 상황에 따라 수정 가능합니다.
    # ...

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature= 0.8
        #stream=True
    )

    assistant_content = response.choices[0].message["content"].strip() 

    send_user_data(transcription)

    
    
    
    print(f"진로 두꺼비: {assistant_content}")



# :modeX: 형식 찾기
    
    #return이 0부터 6으로 나옴

    mode_pattern = r":mode(\d):"
    mode_matches = re.findall(mode_pattern, assistant_content)
    
    for mode in mode_matches:
        print("찾은 mode:", mode)


    # 말에서:mode0: 부터 :mode6: 까지 제거
    for i in range(7):  # 0부터 6까지
        assistant_content = assistant_content.replace(f":mode{i}:", "")
    
    send_assistance_data(assistant_content)



    return assistant_content 





if __name__ == "__main__":
    #record_audio()
    transcription = transcribe_audio("/home/jongil/whisper/recorded_audio.wav")
    print("말", transcription)
    

    if transcription == "":
    	args = ['python3', 'tlqkf_speak.py', '잘 인식되지 않았어요. 다시 말씀해주시겠어요?']
    else:
    	args = ['python3', 'tlqkf_speak.py', converse_with_model(transcription)]
    sbp.Popen(args)

