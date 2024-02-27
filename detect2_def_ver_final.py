import cv2
import numpy as np
from picamera2 import Picamera2
import threading
from queue import Queue
import time
from gpiozero import LED, Buzzer

def configure_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (800,600)}))
    picam2.start()
    return picam2

def detect_objects(frame, net, classes, output_layers, colors, led_q, buzzer_q):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.flip(frame,-1)
    frame = cv2.pyrDown(frame)
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (width, height))

    # ROI 설정
    x = 50; y = 0
    roi_1 = frame[y:y+height, x:x+(width-100)] # 50~350
    cv2.rectangle(roi_1, (0,0), ((width-100)-1,height-1), (255,0,0), 1)
    x = 150; y = 0
    roi_2 = frame[y:y+height, x:x+(width-300)] # 150~250
    cv2.rectangle(roi_2, (0,0), ((width-300)-3,height-1), (0,0,255), 1)

    # 물체 감지
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False) # bolb 객체로 처리(정규화)
    net.setInput(blob) # 생성된 blob 객체에 setInput 함수 적용
    # setInput : YOLO 알고리즘 구현한 딥러닝 모델 초기화 및 입력 데이터 설정 역할
    # 이미지 전처리 > 입력 설정 > 출력 레이저 설정
    outs = net.forward(output_layers) # output_layers을 네트워크 순방향으로 실행(추론)

    # 정보 표시
    class_ids = [] # 인식한 객체 클래스 아이디를 넣는 배열
    confidences = [] # 0~1까지의 객체 인식에 대한 신뢰도를 넣는 배열
    boxes = [] # 객체 인식 후 그릴 상자에 대한 배열

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) # scores 값 중 최대값
            confidence = scores[class_id] # scores 중 class_id에 해당하는 값
            if confidence > 0.5: # 정확도가 0.5 초과 시 인식되었다고 판단
                class_ids.append(class_id)
                if class_ids[0] == 0: # coco.names : person만 인식
                    # 객체 탐지
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                        
                    cv2.circle(frame, (center_x, center_y), 2, (0,0,255), -1)
                    if ((x > 50 and x < 150) or (x > 250 and x < 350)): # ROI 1 구간 내 객체 인식 시 수행
                        # led 점등
                        led_q.put('led1') # led 점등 Queue
                        buzzer_q.put('buzzer1') # buzzer Queue(off)
                        print('ROI 1 Detecting')
                    elif (x >= 150 and x <= 250): # ROI 2 구간 내 객체 인식 시 수행
                        # buzzer 울리기
                        buzzer_q.put('buzzer2') # buzzer Queue(on)
                        print('ROI 2 Detecting')
                    else:
                        led_q.put('None') 
                        buzzer_q.put('None')
                        print('NO Detecting-----1')
                elif len(class_ids) == 0:
                    led_q.put('None') 
                    buzzer_q.put('None')
                    print('NO Detecting---------2')

    # 노이즈 제거 : 같은 사물에 대해 박스가 겹칠 수 있으니, 여러 개인 박스를 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # 인식 객체 label 및 box 설정
    for i in range(len(boxes)):
        if i in indexes:
           if classes[class_ids[i]] == 'person': # 인식한 객체가 person일 시 수행
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    return frame

# led 점등 thread
def led(led_q):
    LED_1 = LED(17)
    LED_2 = LED(27)
    while True:
        led = led_q.get()
        if led == 'led1':
            LED_1.on()
            LED_2.on()
            print('LED ON')
        elif led == 'None':
            LED_1.off()
            time.sleep(0.001)
            LED_2.off()
            time.sleep(0.001)
            print('ALL LED OFF')

# buzzer thread
def buzzer(buzzer_q):
    a_buzzer = Buzzer(22)
    while True:
        buzzer = buzzer_q.get()
        if buzzer == 'buzzer2':
            a_buzzer.on()
            print('Buzzer ON')
        elif buzzer == 'None' or buzzer == 'buzzer1':
            a_buzzer.off()
            time.sleep(0.001)
            print('Buzzer OFF')
            
def main():
    # YOLO 로드
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # YOLO 네트워크와 연결
    classes = []
    with open("coco.names", "r") as f: # coco.names 불러와서 공백 제거 후, classes 배열에 넣음
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames() # 네트워크 모든 레이어 이름 가져와서 layer_names에 넣음
    # 레이어 중 출력 레이어의 인덱스를 가져와 output_layers에 넣음
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # 레이어 중 출력 레이어의 인덱스를 가져와 output_layers에 넣음
    colors = np.random.uniform(0,255,size=(len(classes), 3))

    # Queue 생성(동작 제어를 위함)
    led_q = Queue()
    buzzer_q = Queue()
    # led 및 buzzer 함수를 실행하는 스레드 생성(args 로 Queue 전달)
    led_thread = threading.Thread(target=led, args=(led_q,))
    buzzer_thread = threading.Thread(target=buzzer, args=(buzzer_q,))
    # 스레드 시작
    led_thread.start()
    buzzer_thread.start()

    picam2 = configure_camera()
    while True:
        frame = picam2.capture_array()
        frame = detect_objects(frame, net, classes, output_layers, colors, led_q, buzzer_q)
    
        cv2.imshow('detect', frame)
        key = cv2.waitKey(25) 
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()