from pathlib import Path
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import os
import random
import numpy as np
import pickle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--sp", type=float, default=0.9)
    args = parser.parse_args()
    return args

'''
def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation
'''

def xml_to_annotation(image_path):
    anno_dir = base_dir / 'Train_Annotations'
    image_name = anno_dir / f'{image_path.stem}.xml'
    with image_name.open(encoding='utf-8') as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        bboxes = []
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            '''
            bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
            '''
            box = (float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text))
            bboxes.append(box)
        bboxes = np.array(bboxes, dtype=np.float32)
        #if not bboxes_ignore:
        #bboxes_ignore = np.zeros((0, 4))
        #labels_ignore = np.zeros((0, ))
        annotation = {
            'filename': str(image_path),
            'width': w,
            'height': h,
            'ann': {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.zeros(bboxes.shape[0], dtype=np.int64),
            }
        }
    return annotation





def STAS2pickle():
    train_dir = base_dir / 'Train_Images'
    base_dir_pkl = base_dir / 'custom'
    base_dir_pkl.mkdir(parents=True, exist_ok=True)
    annotations = []
    for image_path in train_dir.iterdir():
        annotations.append(xml_to_annotation(image_path))
    test_annotations = []
    test_dir = base_dir / 'Test_Images'
    for image_path in test_dir.iterdir():
        test_annotations.append({'filename': str(image_path)})
    random.shuffle(annotations)
    print('num of all data: ', len(annotations))
    train_annotations = annotations[:int(len(annotations) * args.sp)]
    print('num of train data: ', len(train_annotations))
    val_annotations = annotations[int(len(annotations) * args.sp):]
    print('num of val data: ', len(val_annotations))
    print('num of test data: ', len(test_annotations))
    All_data_path = base_dir_pkl / 'All_data.pkl'
    with All_data_path.open('wb') as f:
        pickle.dump(annotations, f)
    train_path = base_dir_pkl / 'train.pkl'
    with train_path.open('wb') as f:
        pickle.dump(train_annotations, f)
    val_path = base_dir_pkl / 'val.pkl'
    with val_path.open('wb') as f:
        pickle.dump(val_annotations, f)
    test_path = base_dir_pkl / 'test.pkl'
    with test_path.open('wb') as f:
        pickle.dump(test_annotations, f)
    print('file is save on: ' +'./'+ str(base_dir_pkl))


if __name__ == '__main__':
    args = parse_args()
    base_dir = Path('data') / 'OBJ_Train_Datasets'
    random.seed(666)
    STAS2pickle()


