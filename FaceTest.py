# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import json
import cv2
import base64


def face_test(src):#加一个参数
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "pPGbKn8kQiW2F0DxJfW01WqUvgFzKnNq"
    secret = "AV6xvA5KdPVEaBBctMy05uBpeBiVaonv"
    # filepath = r"D:\Past\smile.jpg"

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    # src = cv2.imread(filepath) #代替成传入图像
    src_encode = cv2.imencode('.jpg', src)[1]
    src_base64 = str(base64.b64encode(src_encode))[2:-1]
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'image_base64')
    # data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(src_base64)
    # data.append('--%s' % boundary)
    # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    # data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append("emotion")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        print("qrcont:",qrcont)
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))
        if len(json.loads(qrcont.decode('utf-8'))['faces']) < 1:
            return None#如果识别不出来脸那就返回这个...
        emotion = json.loads(qrcont.decode('utf-8'))['faces'][0]['attributes']['emotion']
        print("emotion:", emotion)
        return emotion
        # print(qrcont.decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
