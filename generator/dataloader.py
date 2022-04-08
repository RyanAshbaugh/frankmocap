from pathlib import Path
# import xml.etree.ElementTree as ET
import pandas as pd
from os import path, listdir, getcwd, makedirs, cpu_count
import argparse
import pickle
from tqdm import tqdm
# from multiprocessing import Pool
# from multiprocessing import Queue, Process, Manager

'''
# register xml namespaces
ET.register_namespace("","http://www.nist.gov/briar/xml/media")
ET.register_namespace("media","http://www.nist.gov/briar/xml/media")
ET.register_namespace("vc","http://www.w3.org/2007/XMLSchema-versioning")
ET.register_namespace("cmn","http://standards.iso.org/iso-iec/39794/-1")
ET.register_namespace("vstd","http://standards.iso.org/iso-iec/30137/-4")
ET.register_namespace("fstd","http://standards.iso.org/iso-iec/39794/-5")
ET.register_namespace("wb","http://standards.iso.org/iso-iec/39794/-16")
ET.register_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")

ns = {"vstd":"http://standards.iso.org/iso-iec/30137/-4",
        "fstd":"http://standards.iso.org/iso-iec/39794/-5"}

def parallelXMLParse( video_path, process_label_df_dict, index, progress_bar ):

    # BiometricModality values
    # whole body, face
    wb_biomodality, face_biomodality = '15', '1'

    biometric_modalities = { face_biomodality: "face", wb_biomodality: "wb" }
    wb_labels_df_columns = [ 'frame_index', 'x', 'y', 'w', 'h' ]
    face_labels_df_columns = [ 'frame_index', 'x', 'y', 'w', 'h','yaw','pitch' ]

    wb_labels_df = pd.DataFrame(columns=wb_labels_df_columns)
    face_labels_df = pd.DataFrame(columns=face_labels_df_columns)

    labels_dict = { wb_biomodality: wb_labels_df, face_biomodality: face_labels_df }

    tree = ET.parse(video_path)
    root = tree.getroot()

    for frame_annot in root.iterfind('.//vstd:FrameAnnotation', ns ):
        frame_index = int( frame_annot.findall('.//vstd:FrameIndex',ns)[0].text )
        for object_annot in frame_annot.iterfind('.//vstd:ObjectAnnotation', ns ):
            modality = object_annot.findall('.//vstd:BiometricModality',ns)[0].text
            x = int( object_annot.findall('.//vstd:x',ns)[0].text )
            y = int( object_annot.findall('.//vstd:y',ns)[0].text )
            w = int( object_annot.findall('.//vstd:boxWidth',ns)[0].text )
            h = int( object_annot.findall('.//vstd:boxHeight',ns)[0].text )
            label_row_dict = { 'frame_index': frame_index,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h }
            if biometric_modalities[modality] == "face":
                yaw = float( object_annot.findall('.//fstd:yawAngleBlock/fstd:angleValue',ns)[0].text )
                pitch = float( object_annot.findall('.//fstd:pitchAngleBlock/fstd:angleValue',ns)[0].text )
                label_row_dict['yaw'] = yaw
                label_row_dict['pitch'] = pitch

            # add the label, whether face or whole body, to the correct dictionary
            labels_dict[modality] = labels_dict[modality].append( label_row_dict, ignore_index = True )

    process_label_df_dict[str(index)] = { 'wb_labels':labels_dict['15'], 'face_labels':labels_dict['1'] }
    progress_bar.put(1)
'''

def errorCallback(e):
    print(e)

def progressBarListener(q, num_iters):
    progress_bar = tqdm(total = num_iters)
    for item in iter(q.get, None):
        progress_bar.update()

class CasiaBSilhouetteDataset():
    def __init__( self, dataset_path, dataset_pickle_path='' ):
        self.dataset_path = dataset_path
        silhouette_df_columns = ['path',
                                  'fname',
                                  'condition',
                                  'set_num',
                                  'subject_id',
                                  'img_number',
                                  'angle']
        self.silhouette_df = pd.DataFrame( columns=silhouette_df_columns )

        self.prepareSilhouetteDataset(dataset_pickle_path)

    def getImgNames( self ):

        self.silhouette_df['path'] = [ ii for ii in Path( self.dataset_path ).rglob('*.png') ]
        self.silhouette_df['fname'] = [ ii.stem for ii in self.silhouette_df['path'] ]

        return

    def getImgMetaData( self ):

        print('Getting image meta data...')
        num_iters = self.silhouette_df.shape[0]
        progress_bar = tqdm(total = num_iters)
        for ii, img_fname in enumerate( self.silhouette_df['fname'] ):

            img_info = str(img_fname).split('-')
            subject_id = img_info[0]
            condition = img_info[1]
            set_num = img_info[2]
            angle = img_info[3]
            img_num = img_info[4]

            self.silhouette_df['condition'][ii] = condition
            self.silhouette_df['set_num'][ii] = set_num
            self.silhouette_df['subject_id'][ii] = subject_id
            self.silhouette_df['img_number'][ii] = img_num
            self.silhouette_df['angle'][ii] = angle
            progress_bar.update()

    def loadVideoLabels( self ):

        pool = Pool(processes=cpu_count()-4)

        #num_videos = 10
        num_videos = len(self.silhouette_df['labels_path'])

        manager = Manager()
        progress_bar_queue = manager.Queue()
        progress_bar_listener = Process(target=progressBarListener, args=(progress_bar_queue,num_videos))
        progress_bar_listener.start()

        # set up the dataframe to be shared between processes
        process_label_df_dict = manager.dict()

        # for each video, parse the corresponding xml
        for ii, video_path in enumerate( self.silhouette_df['labels_path'][0:num_videos] ):

            # parallel
            pool.apply_async( parallelXMLParse,args=[video_path, process_label_df_dict, ii, progress_bar_queue],\
                error_callback=errorCallback )

        pool.close()
        pool.join()

        progress_bar_queue.put(None)
        progress_bar_listener.join()

        for key,val in process_label_df_dict.items():
            self.silhouette_df['wb_labels'][int(key)] = val['wb_labels']
            self.silhouette_df['face_labels'][int(key)] = val['face_labels']

    def prepareSilhouetteDataset( self, dataset_pickle_path ):

        if path.exists(dataset_pickle_path):
            with open(dataset_pickle_path,'rb') as pkl_file:
                self.silhouette_df = pickle.load( pkl_file )
                pkl_file.close()
        else:
            self.getImgNames()
            self.getImgMetaData()
            #self.loadVideoLabels()
            #briar_pickle_fname = path.join(getcwd(),'briar_dataset.pkl')
            if not path.exists(dataset_pickle_path):
                briar_pickle = open(dataset_pickle_path, 'wb')
                pickle.dump(self.silhouette_df, briar_pickle)
                briar_pickle.close()
        return



