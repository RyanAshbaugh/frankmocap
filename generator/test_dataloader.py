from dataloader import CasiaBSilhouetteDataset

casia_dataset_path = '/home/ryan/iprobe/datasets/casiaB/silhouettes/'
casia_dataset_pickle_path = '/home/ryan/iprobe/datasets/casiaB_silhouettes.pkl'
dataset = CasiaBSilhouetteDataset(casia_dataset_path,
                                  casia_dataset_pickle_path)


