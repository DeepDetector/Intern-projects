import os
from tqdm import tqdm
from shutil import copy

if __name__ == '__main__':
    rootdir = '/raid/huayan_nfs/share/data/detection/online_data'
    targetdir = '/raid/huayan_nfs/share/data/split_data_prelabel'
    dirs = ['20211207_online_train', '20211213_RJ_multi_sku_train', '20211216_online_train', '20220210_online_train',
            '20220428_outsourcing_labeled_refine_train']
    imgindex = 0
    dirindex = 0
    num_perdir = 300

    for dir in dirs:
        dirname = os.path.join(rootdir, dir, 'train', 'JPEGImages')
        print(dirname)
        imgs = os.listdir(dirname)
        num_imgs = len(imgs)
        num_dir = int(num_imgs / num_perdir)
        for i in tqdm(range(0, num_dir)):
            todir = os.path.join(targetdir, dir, "{0:04d}".format(dirindex))
            # print('make dir:' + todir)
            os.makedirs(todir)
            dirindex += 1
            for k in range(0, num_perdir):
                img = imgs[k + i * num_perdir]
                from_path = os.path.join(dirname, img)
                to_path = os.path.join(todir, img)
                # print('from '+from_path+' to '+to_path)
                copy(from_path, to_path)

        todir = os.path.join(targetdir, dir, "{0:04d}".format(dirindex))
        # print('make dir:' + todir)
        os.makedirs(todir)
        dirindex += 1
        for k in range(num_dir * num_perdir, num_imgs):
            img = imgs[k]
            from_path = os.path.join(dirname, img)
            to_path = os.path.join(todir, img)
            # print('from '+from_path+' to '+to_path)
            copy(from_path, to_path)
