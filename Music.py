import threading

from app import db
# import app
# from pose import db
from datetime import *

mutex = threading.Lock()

DB = db

class Music(db.Model):
    __tablename__ = 'music'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime)
    anger = db.Column(db.Integer)
    disgust = db.Column(db.Integer)
    fear = db.Column(db.Integer)
    happiness = db.Column(db.Integer)
    neutral = db.Column(db.Integer)
    sadness = db.Column(db.Integer)
    surprise = db.Column(db.Integer)

    def __repr__(self):
        return '<Music> {}'.format(self.name)


def get_musics():
    return Music.query.filter()


def find_by_id(id):
    return Music.query.filter(Music.id == id)[0]


def insert_music(mmusic):
    db.session.query(Music).filter_by(id=id).update({'time': mmusic.time, "anger": mmusic.anger,
                                                     "disgust": mmusic.disgust,
                                                     "fear": mmusic.fear,
                                                     "happiness": mmusic.happiness,
                                                     "neutral": mmusic.neutral,
                                                     "sadness": mmusic.sadness,
                                                     "surprise": mmusic.surprise})
    print("insert done")
    return -1
    # db.session.add(music)
    # db.session.commit()
    # print("before insert")

    # mutex.acquire()

    # db.session.add(music)
    # db.session.update()
    # db.session.commit()
    # Id = int(music.id)
    # print("music_id:", Id)
    # db.drop_all()

    db.create_all()
    # try:
    #     old = Music.query.filter_by(id=mmusic.id)[0]
    #     # print("after filter")
    #     # # 更新用户的数据
    #     old.time = mmusic.time
    #     old.anger = mmusic.anger
    #     old.disgust = mmusic.disgust
    #     old.fear = mmusic.fear
    #     old.happiness = mmusic.happiness
    #     old.neutral = mmusic.neutral
    #     old.sadness = mmusic.sadness
    #     old.surprise = mmusic.surprise
    #     # # 最后提交到数据库
    #     db.session.add(old)
    #     print("before commit")
    #     # db.session.commit()
    #     # db.session.flush()
    #     print("after commit")
    # except Exception as e:
    #     # 加入数据库commit提交失败，必须回滚！！！
    #     db.session.rollback()
    #     raise e
    # return -1

    old = Music.query.filter_by(id=mmusic.id)[0]
    # print("after filter")
    # # 更新用户的数据
    old.time = mmusic.time
    old.anger = mmusic.anger
    old.disgust = mmusic.disgust
    old.fear = mmusic.fear
    old.happiness = mmusic.happiness
    old.neutral = mmusic.neutral
    old.sadness = mmusic.sadness
    old.surprise = mmusic.surprise
    # # 最后提交到数据库
    db.session.add(old)
    print("before commit")
    db.session.commit()
    print("after commit")
    return -1

    Id = int(mmusic.id)
    print("music_id:", Id)
    # # db.session.query(Music).filter_by(id=Id).update({
    # DB.session.query(Music).filter(Music.id == Id)
    # print("find")
    DB.session.query(Music).filter(Music.id == Id).update({
        'time': mmusic.time,
        'anger': int(mmusic.anger),
        'disgust': int(mmusic.disgust),
        'fear': int(mmusic.fear),
        'happiness': int(mmusic.happiness),
        'neutral': int(mmusic.neutral),
        'sadness': int(mmusic.sadness),
        'surprise': int(mmusic.surprise)
    })
    # mutex.release()


if __name__ == '__main__':
    # db.drop_all()
    # db.create_all()

    time = datetime.now()

    musics = get_musics()
    music = find_by_id(1926723)
    print(music.surprise)
    # musics[0].anger = 0  # 第一首歌
    # insert_music(musics[0])  # 也就是修改了属性之后的样子。
    music.anger = 0  # 第一首歌
    insert_music(music)  # 也就是修改了属性之后的样子。
