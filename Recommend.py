import numpy as np
from FaceTest import face_test
from Music import get_musics, insert_music
from datetime import *


class Recommend:
    def __init__(self):
        self.musics = get_musics()

    def recommend(self, emotion):
        if emotion==None: #识别不出来脸
            return -1
        face_emotion = np.zeros(7)
        face_emotion[0] = emotion['anger']
        face_emotion[1] = emotion['disgust']
        face_emotion[2] = emotion['fear']
        face_emotion[3] = emotion['happiness']
        face_emotion[4] = emotion['neutral']
        face_emotion[5] = emotion['sadness']
        face_emotion[6] = emotion['surprise']

        musics = self.musics
        max_value = 0
        max_idx = 0
        for i in range(musics.count()):
            music_emotion = np.zeros(7)
            music_emotion[0] = musics[i].anger
            music_emotion[1] = musics[i].disgust
            music_emotion[2] = musics[i].fear
            music_emotion[3] = musics[i].happiness
            music_emotion[4] = musics[i].neutral
            music_emotion[5] = musics[i].sadness
            music_emotion[6] = musics[i].surprise

            emotion_value = face_emotion.dot(music_emotion)

            delta_time = datetime.now() - musics[i].time
            interval_time = 24 * 60 * 60
            if delta_time.total_seconds() < interval_time:
                emotion_value *= (delta_time.total_seconds() / interval_time)

            if emotion_value > max_value:
                max_value = emotion_value
                max_idx = i

        musics[max_idx].time = datetime.now()
        insert_music(musics[max_idx])

        print(max_value)

        return musics[max_idx].id


# if __name__ == '__main__':
#     rec = Recommend()
#     rec.recommend(face_test())
