import cv2
import pyttsx3
import requests
engine=pyttsx3.init()
# ESP32 URL
URL = "http://192.168.139.245"
AWB = True
cap = cv2.VideoCapture(URL + ":81/stream")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem
def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))
        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")
def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))

    except:
        print("SET_QUALITY: something went wrong")
    return awb
if __name__ == '__main__':
    set_resolution(URL, index=8)

# recognition and opencv setup
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'
model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels=[]
file_name='dataset.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


#for video detection

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")

font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
i=1
while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)

    if(i%30==0):
        print(ClassIndex)
        for i in ClassIndex:
            if(i==1):
                continue

                engine.say("A person is infront of your house")
                engine.runAndWait()
            if(i==17):
                engine.say("warning, a cat is detected")
                engine.runAndWait()
            if(i==18):
                engine.say("A dog has been detected")
                engine.runAndWait()
            if(i==19):
                engine.say("a horse is detected")
                engine.runAndWait()
            if(i==20):
                engine.say("sheep is detected")
                engine.runAndWait()
            if(i==22):
                engine.say("alert,an elephant is detected")
                engine.runAndWait()

    if(len(ClassIndex)!=0):
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0,200,0),thickness=5)
    cv2.imshow('object detection tutorial',frame)
    i+=1
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindow()