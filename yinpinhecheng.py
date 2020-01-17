from pydub import AudioSegment
import os
import shutil
import time

# import sort

AudioSegment.converter = "D:\\Program Files (x86)\\ffmpeg-4.1-win64-static\\bin\\ffmpeg.exe"

now_dir = os.getcwd()  # 当前文件夹
new_dir = now_dir + '\\voicepcm\\'  # 语音文件所在文件夹
list_voice_dir = os.listdir('./voicepcm')
list_voice_dir.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大


def voice_unit():
    n = 0

    # list_voice_dir_length = len(list_voice_dir)
    playlist = AudioSegment.empty()
    second_5_silence = AudioSegment.silent(duration=5000)  # 产生一个持续时间为5s的无声AudioSegment对象
    for i in list_voice_dir:
        # sound = AudioSegment.from_wav(list_voice_dir[n])
        # sound = AudioSegment.from_file(new_dir+list_voice_dir[n],format="wav") #  wav
        # raw pcm  # 2 byte (16 bit) samples 采样sample， channels = 1为单声道，2为立体声
        sound = AudioSegment.from_file(new_dir + list_voice_dir[n], sample_width=2, frame_rate=16000, channels=1)
        playlist = playlist + sound + second_5_silence

        n += 1
    # playlist.export(new_dir+'playlist.pcm',format="pcm") #wav
    playlist.export(new_dir + 'playlist100.pcm')
    print("语音合成完成，合成文件放在：", new_dir, "目录下")


def testlist():
    n = 0
    for i in list_voice_dir:
        voicename = list_voice_dir[n]
        print("对比文件顺序是否改变：", voicename)
        n += 1


def main():
    try:
        os.remove(new_dir + 'playlist.pcm')
    except:
        print("")
    testlist()
    voice_unit()


if __name__ == "__main__":
    main()
