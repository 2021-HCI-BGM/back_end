from app import db
# from pose import db
from datetime import *


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


def insert_music(music):
    db.session.add(music)
    db.session.commit()


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
