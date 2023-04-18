# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from sklearn.model_selection import StratifiedKFold, KFold


def create_kfold_split_uniform(folds, seed=42):
    cache_path = OUTPUT_PATH + 'kfold_split_{}_{}.csv'.format(folds, seed)
    if not os.path.isfile(cache_path):
        kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
        files = glob.glob(INPUT_PATH + 'frames_50_video/*.mp4')
        names = [os.path.basename(f) for f in files]

        s = pd.DataFrame(names, columns=['video_id'])
        s['fold'] = -1
        for i, (train_index, test_index) in enumerate(kf.split(s.index)):
            s.loc[test_index, 'fold'] = i
        s.to_csv(cache_path, index=False)
        print('No folds: {}'.format(len(s[s['fold'] == -1])))

        for i in range(folds):
            part = s[s['fold'] == i]
            print(i, len(part))
    else:
        print('File already exists: {}'.format(cache_path))


if __name__ == '__main__':
    create_kfold_split_uniform(5, seed=42)
