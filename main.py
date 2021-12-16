import os
import cv2
import argparse
import numpy as np
import OpticalFlow.OpticalFlow as of

parser = argparse.ArgumentParser(description='オプティカルフロー')
parser.add_argument('--movie_path', help='動画ファイルパス', required=True)
parser.add_argument('--model', help='使用モデル。\n lucas: lucas_kanade法\n farneback: farneback法', required=True)
parser.add_argument('--show', help='画面表示', action='store_true')
parser.add_argument('--output_dir_path', help='出力画像の保存ディレクトリパス')

args = parser.parse_args()


def main(video_path, model='lucas'):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    opticalflow = of.OpticalFlow(old_gray)
    if model == 'lucas':
        mask = np.zeros_like(old_frame)
    elif model == 'farneback':
        mask = np.zeros_like(old_frame)
        mask[..., 1] = 255
    else:
        print('不正なモデル')
        return

    count = 0
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if model == 'lucas':
            img = opticalflow.lucas_kanade(frame, mask, old_gray, frame_gray)
        else:
            img = opticalflow.farneback(mask, old_gray, frame_gray)

        old_gray = frame_gray.copy()

        if args.show:
            cv2.imshow('test', img)
            cv2.waitKey(1)

        if args.output_dir_path:
            os.makedirs(args.output_dir_path, exist_ok=True)
            cv2.imwrite(f'{args.output_dir_path}/{str(count).zfill(16)}.jpg', img)
            count = count+1

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main(args.movie_path, args.model)