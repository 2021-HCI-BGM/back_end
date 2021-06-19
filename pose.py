# from json.decoder import JSONObject
import base64
import os
import re

import cv2 as cv
import numpy as np
import argparse
from threading import Lock, Thread
from flask_socketio import Namespace, emit, SocketIO, disconnect
from flask import Flask, request
import preProcess
import cv2
from flask_sqlalchemy import SQLAlchemy
from FaceTest import face_test
from Music import get_musics, insert_music, find_by_id
from datetime import *
from Recommend import Recommend
import json
from enum import Enum

import inspect
import ctypes


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def json_to_object(data):
    return json.loads(data, object_hook=lambda d: Namespace(**d))


def find_most(musics):
    music_emotion = np.zeros(7)
    music_emotion[0] = musics.anger
    music_emotion[1] = musics.disgust
    music_emotion[2] = musics.fear
    music_emotion[3] = musics.happiness
    music_emotion[4] = musics.neutral
    music_emotion[5] = musics.sadness
    music_emotion[6] = musics.surprise
    max_value = 0
    max_idx = 0
    for i in range(7):  # 0~6
        if music_emotion[i] > max_value:
            max_value = music_emotion[i]
            max_idx = i
    return max_idx  # 返回的是一个0~6的，然后再根据这个去修改判断...


def change_score(music, emotion, index, percent):
    loss = 0
    if index == 0:
        loss = music.anger * (1 - percent)
        music.anger = music.anger * percent
    elif index == 1:
        loss = music.disgust * (1 - percent)
        music.disgust = music.disgust * percent
    elif index == 2:
        loss = music.fear * (1 - percent)
        music.fear = music.fear * percent
    elif index == 3:
        loss = music.happiness * (1 - percent)
        music.happiness = music.happiness * percent
    elif index == 4:
        loss = music.neutral * (1 - percent)
        music.neutral = music.neutral * percent
    elif index == 5:
        loss = music.sadness * (1 - percent)
        music.sadness = music.sadness * percent
    elif index == 6:
        loss = music.surprise * (1 - percent)
        music.surprise = music.surprise * percent
    # 然后把对应的加到我们选择的感情上面去
    if emotion == "anger":
        music.anger += loss
    elif emotion == "disgust":
        music.disgust += loss
    elif emotion == "fear":
        music.fear += loss
    elif emotion == "hapiness":
        music.hapiness += loss
    elif emotion == "neutral":
        music.neutral += loss
    elif emotion == "sadness":
        music.sadness += loss
    elif emotion == "surprise":
        music.surprise += loss
    return music


def strengthen(emotion):
    if emotion is None:  # 识别不出来脸
        return None
    face_emotion = np.zeros(7)
    face_emotion[0] = emotion['anger']
    face_emotion[1] = emotion['disgust']
    face_emotion[2] = emotion['fear']
    face_emotion[3] = emotion['happiness']
    face_emotion[4] = emotion['neutral']
    face_emotion[5] = emotion['sadness']
    face_emotion[6] = emotion['surprise']
    max_value = 0
    max_idx = 0
    for i in range(7):  # 0~6
        if face_emotion[i] > max_value:
            max_value = face_emotion[i]
            max_idx = i
    if max_idx == 0:
        emotion['anger'] = emotion['anger'] * 1.5
    elif max_idx == 1:
        emotion['disgust'] = emotion['disgust'] * 1.5
    elif max_idx == 2:
        emotion['fear'] = emotion['fear'] * 1.5
    elif max_idx == 3:
        emotion['happiness'] = emotion['happiness'] * 1.5
    elif max_idx == 4:
        emotion['neutral'] = emotion['neutral'] * 1.5
    elif max_idx == 5:
        emotion['sadness'] = emotion['sadness'] * 1.5
    elif max_idx == 6:
        emotion['surprise'] = emotion['surprise'] * 1.5
    return emotion


class PoseParser:
    def __init__(self, pose):
        self.pose = pose

    def Parse(self, frame):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

        args = parser.parse_args()

        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        inWidth = args.width
        inHeight = args.height

        net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

        # cap = cv.VideoCapture(args.input if args.input else 0)
        #
        # while cv.waitKey(1) < 0:
        #     hasFrame, frame = cap.read()
        #     if not hasFrame:
        #         cv.waitKey()
        #         break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(
            cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        pose = "NORMAL"

        if points[15] and points[14]:
            if points[15][1] > points[14][1]:
                pose = "LEFT_TILT"

            if points[15][1] < points[14][1]:
                pose = "RIGHT_TILT"

        if points[7] and points[6] and points[5] and points[4] and points[3] and points[2]:
            if points[7][1] > points[6][1] > points[5][1]:
                pose = "LEFT_ARM_RISE"
                if points[4][1] > points[3][1] > points[2][1]:
                    pose = "BOTH_ARM_RISE"
            elif points[4][1] > points[3][1] > points[2][1]:
                pose = "LEFT_ARM_RISE"

        body_pose_tag = "Pose: "
        if self.pose != pose:
            self.pose = pose
            # print("Pose:", pose)

        cv.putText(frame, body_pose_tag + pose, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # cv.imshow('OpenPose using OpenCV', frame)


# HandParse
flag = np.array([-1, -1])


class general_pose_model(object):
    def __init__(self, modelpath):
        self.num_points = 22
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                            [0, 5], [5, 6], [6, 7], [7, 8],
                            [0, 9], [9, 10], [10, 11], [11, 12],
                            [0, 13], [13, 14], [14, 15], [15, 16],
                            [0, 17], [17, 18], [18, 19], [19, 20]]
        # self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.2
        self.hand_net = self.get_hand_model(modelpath)
        self.gesture = "Not Defined"

    def get_hand_model(self, modelpath):

        prototxt = os.path.join(modelpath, "hand/pose_deploy.prototxt")
        caffemodel = os.path.join(modelpath, "hand/pose_iter_102000.caffemodel")
        hand_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return hand_model

    def predict(self, img_cv2):
        # img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        aspect_ratio = img_width / img_height

        inWidth = int(((aspect_ratio * self.inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(img_cv2, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False,
                                        crop=False)
        self.hand_net.setInput(inpBlob)
        output = self.hand_net.forward()
        # vis heatmaps
        # self.vis_heatmaps(imgfile, output)

        #
        points = []
        limit_points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :]  # confidence map.
            probMap = cv2.resize(probMap, (img_width, img_height))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                # points.append((int(point[0]), int(point[1])))
                cv2.circle(img_cv2, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(img_cv2, "{}".format(idx), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (0, 0, 255), 2, lineType=cv2.LINE_AA)
                points.append(np.array([int(point[0]), int(point[1])]))

            else:
                # points.append(None)
                points.append(flag)

        # 手势识别部分：
        limit = np.concatenate(([points[0]], [points[3]], [points[7]], [points[11]], [points[15]], [points[19]]),
                               axis=0)
        # print(limit)
        for i in limit:
            if (i == flag).any():
                pass
            else:
                limit_points.append(i)
        limit_points = np.array(limit_points)

        if limit_points.size > 0:
            (x, y), radius = cv2.minEnclosingCircle(limit_points)
            # print(x,y,radius)
            center = (int(x), int(y))
            radius = int(radius)
            flag_distance = radius * 0.1
            # print(points[4],points[8],points[12],points[16],points[20])
            # 计算每个判断点到圆边界的距离
            if (points[4] != flag).all():
                distance_4 = np.linalg.norm(points[4] - center) - radius
            else:
                distance_4 = 0
            if (points[8] != flag).all():
                distance_8 = np.linalg.norm(points[8] - center) - radius
            else:
                distance_8 = 0
            if (points[12] != flag).all():
                distance_12 = np.linalg.norm(points[12] - center) - radius
            else:
                distance_12 = 0
            if (points[16] != flag).all():
                distance_16 = np.linalg.norm(points[16] - center) - radius
            else:
                distance_16 = 0
            if (points[20] != flag).all():
                distance_20 = np.linalg.norm(points[20] - center) - radius
            else:
                distance_20 = 0

            # if distance_8 >= flag_distance and (
            #         np.array([distance_4, distance_12, distance_16, distance_20]) < flag_distance).all():
            #     result = "this is one"
            #     self.gesture = "ONE"
            # elif (np.array([distance_8, distance_12]) >= flag_distance).all() and (
            #         np.array([distance_4, distance_16, distance_20]) < flag_distance).all():
            #     result = "this is two"
            # elif (np.array([distance_8, distance_12, distance_16,
            #                 distance_20]) >= flag_distance).all() and distance_4 < flag_distance:
            #     result = "this is four"
            # elif (np.array([distance_4, distance_8, distance_12, distance_16, distance_20]) >= flag_distance).all():
            #     result = "this is five"
            # elif (np.array([distance_4, distance_8, distance_12, distance_16, distance_20]) <= flag_distance).all():
            #     result = "this is fist"
            # elif (np.array([distance_4, distance_8]) >= flag_distance).all() and (
            #         np.array([distance_12, distance_16, distance_20]) < flag_distance).all():
            #     result = "this is eight"
            if distance_4 >= flag_distance and (
                    np.array([distance_8, distance_16, distance_20]) < flag_distance).all():
                result = "this is good"
                self.gesture = "GOOD"
            # elif distance_12 >= flag_distance and (
            #         np.array([distance_4, distance_8, distance_16, distance_20]) < flag_distance).all():
            #     result = "this is out"
            elif (np.array([distance_12, distance_16]) >= flag_distance).all() and distance_4 < flag_distance:
                result = "this is OK"
                self.gesture = "OK"

            # elif (np.array([distance_16, distance_20]) >= flag_distance).all() and distance_4 < flag_distance:
            #     result = "this is love"
            else:
                result = "can't find"
                self.gesture = "Not Defined"

            cv2.putText(img_cv2, result, (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        return points

    # def vis_heatmaps(self, imgfile, net_outputs):
    #     img_cv2 = cv2.imread(imgfile)
    #     plt.figure(figsize=[10, 10])
    #
    #     for pdx in range(self.num_points):
    #         probMap = net_outputs[0, pdx, :, :]
    #         probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
    #         plt.subplot(5, 5, pdx + 1)
    #         plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    #         plt.imshow(probMap, alpha=0.6)
    #         plt.colorbar()
    #         plt.axis("off")
    #     plt.show()

    def vis_pose(self, img_cv2, points):
        # img_cv2 = cv2.imread(imgfile)
        img_cv2_copy = np.copy(img_cv2)
        # for idx in range(len(points)):
        #     if points[idx]:
        #         cv2.circle(img_cv2_copy, points[idx], 8, (0, 255, 255), thickness=-1,
        #                    lineType=cv2.FILLED)
        #         cv2.putText(img_cv2_copy, "{}".format(idx), points[idx], cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Draw Skeleton
        for pair in self.point_pairs:
            partA = pair[0]
            partB = pair[1]

            if (points[partA] != flag).any() and (points[partB] != flag).any():
                cv2.line(img_cv2, tuple(points[partA]), tuple(points[partB]), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(img_cv2, tuple(points[partA]), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(img_cv2, tuple(points[partB]), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # cv2.imshow('OpenPose using OpenCV', img_cv2)


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@localhost:3306/music"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

db = SQLAlchemy(app)

thread = None


def testThreadFunction():
    parser = PoseParser("NORMAL")
    cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
    tmp_pose = "NORMAL"
    tmp_gesture = "Not Defined"
    modelpath = "./model/"
    pose_model = general_pose_model(modelpath)  # 读取手势识别模型
    startorpause = False
    rec = Recommend()
    while cv.waitKey(1) < 0:
        print("time:", startorpause)
        hasFrame, frame = cap.read()
        if not hasFrame:  # 如果读取不正确
            cv.waitKey()
            cap = cv2.VideoCapture(0)
            print("Not Ok")
            continue
        print("pre:", startorpause)
        pre = preProcess.preprocess(frame)  # 预处理

        parser.Parse(frame)  # 身体姿势分析
        res_points = pose_model.predict(frame)  # 手势分析
        if pose_model.gesture != tmp_gesture and pose_model.gesture != "Not Defined":  # 检测到手势变化
            tmp_gesture = pose_model.gesture
            print(tmp_gesture)
            if tmp_gesture == "GOOD":
                startorpause = True  # 此时识别出是trigger动作
            if tmp_gesture == "OK":
                startorpause = False  # 此时识别出是关闭动作
            # 如果是其他动作或者检测不到，那么就还是维持上一循环的数据
            socketio.emit('trigger', startorpause, namespace='/test')

        if parser.pose != tmp_pose:  # 当pose发生变化
            tmp_pose = parser.pose
            print(parser.pose)
            # socketio.emit('server_response', {'data': tmp_pose, 'time': 'now'}, namespace='/test')
            # 在emit之前需要加一层判断，判断得到的pose是开始的trigger还是结束的trigger
            # if tmp_pose == "LEFT_TILT":
            #     startorpause = True  # 此时识别出是trigger动作
            # if tmp_pose == "RIGHT_TILT":
            #     startorpause = False  # 此时识别出是关闭动作
            # # 如果是其他动作或者检测不到，那么就还是维持上一循环的数据
            # socketio.emit('trigger', startorpause, namespace='/test')
        # print("time:", "now")
        # 这里还有一个逻辑，如果trigger是false的话，也就停止后面的识别动作了...
        print("judge:", startorpause)
        if startorpause == False:
            continue
        # 只有是true的时候才识别，也就是加上吕泽宇和数据库的操作

        # 可以考虑加一个计数器来降低更新频率？

        # 根据手势对情感进行一个强化
        if tmp_pose == "BOTH_ARM_RISE" or tmp_pose == "LEFT_ARM_RISE" or tmp_pose == "RIGHT_ARM_RISE":  # 把手举高作为一个强化的标志
            emotion = strengthen(face_test(frame))
        else:
            emotion = face_test(frame)
        rec_id = rec.recommend(emotion)  # 得到歌曲的id
        # rec_id = rec.recommend(face_test(frame))  # 得到歌曲的id
        print("rec_id", rec_id)
        if rec_id != -1:  # 其实也可以前端去判断？
            socketio.emit('url', rec_id, namespace='/test')

    cap.release()


def decode_base64(data):
    """Decode base64, padding being optional.
    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += b'=' * missing_padding
    return base64.decodestring(data)


def base64_to_image(base64_code):
    # base64_code = base64_code.replace(' ', '+')
    # base64解码
    # base64_code=decode_base64(base64_code)
    # base64_code=base64.decodestring(base64_code.encode('ascii'))
    # img_data = base64.b64decode(base64_code)
    #
    # # img_data = base64_code
    # # 转换为np数组
    # img_array = np.fromstring(str(img_data), np.uint8)
    # # 转换成opencv可用格式
    # # img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("test:", base64_code)
    imgData = base64.b64decode(base64_code)
    nparr = np.frombuffer(imgData, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imshow("test",img_np)
    # cv2.waitKey(0)
    return img


class Listener(Namespace):
    def __init__(self, namespace):
        super(Listener, self).__init__(namespace)
        self.sid = None

    def on_connect(self):  # 这里的名字可能要改一下——改成begin，begin之后就开启线程（？那么关闭？on_close?）
        self.sid = request.sid
        print('建立连接成功！-{}'.format(self.sid))
        # 连接之后，在这里新开一个线程，跑图像处理的部分
        # 注意：断开连接后，需要将线程停止
        emit('server_response', {'data': 'Connected', 'count': 0})  # 返回消息给前端告诉他连接成功了
        # global thread
        # if thread is None:
        #     thread = Thread(target=testThreadFunction)
        #     thread.daemon = True  # 守护？可以自动结束子进程？
        #     print("创建线程")
        #     thread.start()
        #     print("开始线程")

    def on_score(self, data):  # 前端评分传给我们后端
        print("score:", data)
        # info=json.loads(data,object_hook=JSONObject)
        info = json_to_object(data)
        print("info:", info)
        id = info.id
        score = info.score
        emotion = info.emotion  # 这里的感情应该是一个字符串
        music = find_by_id(id)
        # 然后根据分数去修改对应的属性...
        index = find_most(music)  # 得到了最大的属性的标号。
        # 然后根据分数来判断
        if score == 5:
            print(5)
            if index == 0:
                music.anger = music.anger * 1.1
            elif index == 1:
                music.disgust = music.disgust * 1.1
            elif index == 2:
                music.fear = music.fear * 1.1
            elif index == 3:
                music.happiness = music.happiness * 1.1
            elif index == 4:
                music.neutral = music.neutral * 1.1
            elif index == 5:
                music.sadness = music.sadness * 1.1
            elif index == 6:
                music.surprise = music.surprise * 1.1
        elif score == 4:
            print(4)
            # do nothing
        elif score == 3:
            print(3)
            # 之后就需要加上感情了
            # 首先最大的感情降低，然后把那些多的加到对应的感情上面去（可以写成函数复用）
            music = change_score(music, emotion, index, 0.8)  # 这里的0.8是指保留0.8 loss还是0.2
        elif score == 2:
            print(2)
            music = change_score(music, emotion, index, 0.7)
        elif score == 1:
            music = change_score(music, emotion, index, 0.5)
            print(1)
        insert_music(music)

    def on_picture(self, data):
        # img_np=cv2.imread("./test.jpg")
        # cv2.imshow("test:", img_np)
        # cv2.waitKey(0)
        # image = cv2.imencode('.jpg', img_np)[1]
        # print("picture:", base64.b64encode(image))
        # base64_data = str(base64.b64encode(image))[2:-1]
        # data = base64_data
        # data = str(data)[2:-1]
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", data, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")

        else:
            raise Exception("Do not parse!")

        # 2、base64解码
        imgData = base64.urlsafe_b64decode(data)
        nparr = np.frombuffer(imgData, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print("picture:", data)
        # img = base64_to_image(data)
        # cv.imshow("img:", img)
        # cv2.waitKey(0)
        frame = img
        parser = PoseParser("NORMAL")
        tmp_pose = "NORMAL"
        tmp_gesture = "Not Defined"
        modelpath = "./model/"
        pose_model = general_pose_model(modelpath)  # 读取手势识别模型
        startorpause = False
        rec = Recommend()
        print("time:", startorpause)
        print("pre:", startorpause)
        pre = preProcess.preprocess(frame)  # 预处理
        parser.Parse(frame)  # 身体姿势分析
        res_points = pose_model.predict(frame)  # 手势分析
        if pose_model.gesture != tmp_gesture and pose_model.gesture != "Not Defined":  # 检测到手势变化
            tmp_gesture = pose_model.gesture
            print("tmp_gesture:",tmp_gesture)
            if tmp_gesture == "GOOD":
                startorpause = True  # 此时识别出是trigger动作
            if tmp_gesture == "OK":
                startorpause = False  # 此时识别出是关闭动作
            # 如果是其他动作或者检测不到，那么就还是维持上一循环的数据
            socketio.emit('trigger', startorpause, namespace='/test')
        if parser.pose != tmp_pose:  # 当pose发生变化
            tmp_pose = parser.pose
            print("parser",parser.pose)
        print("judge:", startorpause)
        if startorpause == False:
            return None
        # 只有是true的时候才识别，也就是加上吕泽宇和数据库的操作

        # 可以考虑加一个计数器来降低更新频率？
        # 根据手势对情感进行一个强化
        if tmp_pose == "BOTH_ARM_RISE" or tmp_pose == "LEFT_ARM_RISE" or tmp_pose == "RIGHT_ARM_RISE":  # 把手举高作为一个强化的标志
            emotion = strengthen(face_test(frame))
        else:
            emotion = face_test(frame)
        rec_id = rec.recommend(emotion)  # 得到歌曲的id
        print("rec_id", rec_id)
        if rec_id != -1:  # 其实也可以前端去判断？
            socketio.emit('url', rec_id, namespace='/test')


def on_disconnect(self):  # 其实可以没有...
    print('客户端断开连接！')
    # 然后关闭线程
    global thread
    # stop_thread(thread)
    thread = None


def close_room(self, room):
    socketio.close_room(room=self.sid)
    print('{}-断开连接'.format(self.sid))


def on_my_event(self, data):  # 这里拿到前端的my_event接口发送的数据data
    print(data)


socketio.on_namespace(Listener('/test'))  # 路径是http://localhost:5000/test

if __name__ == '__main__':
    # parser = PoseParser("NORMAL")
    socketio.run(app)
    # parser.Parse()
