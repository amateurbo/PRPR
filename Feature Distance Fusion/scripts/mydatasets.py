from torchreid.data import ImageDataset
import os
import glob
import os.path as osp

class MTA_reid(ImageDataset):
    dataset_dir = 'MTA_reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        super(MTA_reid, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        imgnames = os.listdir(dir_path)
        pid_con = set()
        for imgname in imgnames:
            pid = int(imgname.split('_')[5].split('.')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        for imgname in imgnames:
            imgpath = osp.join(dir_path, imgname)
            pid = int(imgname.split('_')[5].split('.')[0]) - 1
            camid = int(imgname.split('_')[3])
            # if((dir_path.split('/')[-1] == 'query')):
            #     camid = camid - 1
            # assert 0 <= camid <= 1
            if relabel:
                pid = pidlabel[pid]
            list.append((imgpath, pid, camid))

        return list


class MTA_reid_new(ImageDataset):
    # dataset_dir = 'MTA_reid_new'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        super(MTA_reid_new, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        imgnames = os.listdir(dir_path)
        pid_con = set()
        for imgname in imgnames:
            # pid = int(imgname.split('_')[5].split('.')[0]) - 1
            pid = int(imgname.split('_')[5].split('x')[0].split('h')[0].split('.')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        for imgname in imgnames:
            imgpath = osp.join(dir_path, imgname)
            # pid = int(imgname.split('_')[5].split('.')[0]) - 1
            pid = int(imgname.split('_')[5].split('x')[0].split('h')[0].split('.')[0]) - 1
            camid = int(imgname.split('_')[3])
            # if((dir_path.split('/')[-1] == 'query')):
            #     camid = camid - 1
            # assert 0 <= camid <= 1
            if relabel:
                pid = pidlabel[pid]
            list.append((imgpath, pid, camid))

        return list

