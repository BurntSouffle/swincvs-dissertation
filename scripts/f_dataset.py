import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
import pandas as pd
import shutil

from torchvision import transforms
from pathlib import Path
from scripts.f_environment import download_extract_zip

def get_datasets(config):
    """
    Check dataset exists (download if not). Create dataset instances and apply transformations specified in config
    """
    dataset_dir = check_dataset(config)
    print(f"\nDataset loaded from: {dataset_dir}")

    # F1a soft-label intervention: if TRAIN.SOFT_TRAIN_LABELS is True in the
    # config, the train split uses the raw mean annotator-agreement values
    # (in {0, 0.333, 0.667, 1.0}) as targets, instead of the MV-rounded {0, 1}.
    # Val and test always stay MV so best-val-mAP selection and test scoring
    # are identical to the baseline. Backward-compatible via hasattr.
    soft_train = (hasattr(config.TRAIN, 'SOFT_TRAIN_LABELS')
                  and config.TRAIN.SOFT_TRAIN_LABELS)
    # F1b disagreement-prediction auxiliary head: if MODEL.F1B_DISAGREEMENT_HEAD
    # is True, every split returns a 6-vector label [C1, C2, C3, D1, D2, D3],
    # where D{1,2,3} = 1 iff the criterion's three annotators were NOT
    # unanimous on that frame (i.e. 0 < ds < 1). The training loop slices
    # [:3] for the MV-head loss and [3:] for the disagreement-head loss.
    include_disagreement = (hasattr(config.MODEL, 'F1B_DISAGREEMENT_HEAD')
                            and config.MODEL.F1B_DISAGREEMENT_HEAD)
    train_dataframe, val_dataframe, test_dataframe = get_three_dataframes(
        dataset_dir, lstm=config.MODEL.LSTM, soft_train=soft_train,
        include_disagreement=include_disagreement)

    transform_sequence = get_transform_sequence(config)

    # If SwinCVS model
    if config.MODEL.LSTM:
            training_dataset = EndoscapesSwinCVS_Dataset(train_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
            val_dataset = EndoscapesSwinCVS_Dataset(val_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
            test_dataset = EndoscapesSwinCVS_Dataset(test_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
    # If just SwinV2 backbone
    else:
            training_dataset = Endoscapes_Dataset(train_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
            val_dataset = Endoscapes_Dataset(val_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
            test_dataset = Endoscapes_Dataset(test_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)

    return training_dataset, val_dataset, test_dataset

def check_dataset(config):
    """
    Checks whether specified folder contains valid endoscapes dataset. Redownloads if checksum failed, or folder missing.
    Requires config.DATASET_DIR to lead to the folder containing 'endoscapes' or null - will download to repo dir.
    """
    dataset_path = config.DATASET_DIR

    # If dataset is meant to be downloaded into cwd
    if dataset_path == None:
        dataset_path = Path.cwd()

    # Add 'endoscapes' subfolder
    dataset_dir = Path(dataset_path) / 'endoscapes'

    # Checksum of the number of expected files
    all_imgs_dir = dataset_dir / 'all'
    if all_imgs_dir.exists() and all_imgs_dir.is_dir():
        file_count = sum(1 for f in all_imgs_dir.iterdir() if f.is_file())
        if file_count != 58586:
            # Non-interactive: warn but continue (avoids hanging on remote runs)
            if os.environ.get("SWINCVS_AUTO", "0") == "1" or not sys.stdin.isatty():
                print(f"WARNING: Dataset checksum failed ({file_count}/58586 files). Continuing anyway.")
            else:
                response = input(f"Dataset checksum failed. Attempting to remove the '{dataset_dir}' and redownload Endoscapes dataset. Proceed? (Y/N): ").strip().upper()
                while response not in ['Y', 'N']:
                    input(f"Please answer with Y/N only")
                if response == 'Y':
                    print('Removing pre-existing dataset...')
                    shutil.rmtree(dataset_dir)
                    print('Re-downloading dataset')
                    download_extract_zip(dataset_dir.parent, 'https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip')
                if response == 'N':
                    print("Continuing with the originally specified dataset...")
    else:
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_dir}. "
                f"Set DATASET_DIR env var or --dataset_dir to the directory containing 'endoscapes/'."
            )
        print(f"WARNING: {all_imgs_dir} not found, but {dataset_dir} exists. Continuing.")
    return dataset_dir

def get_dataloaders(config, training_dataset, val_dataset, test_dataset):
    """
    Create dataloaders from a given training datasets
    """
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}, num_workers: {num_workers}")
    train_dataloader = DataLoader(  training_dataset,
                                    batch_size = config.TRAIN.BATCH_SIZE,
                                    pin_memory = True,
                                    shuffle = True,
                                    num_workers = num_workers)

    val_dataloader = DataLoader(    val_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True,
                                    num_workers = num_workers)

    test_dataloader = DataLoader(   test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True,
                                    num_workers = num_workers)
    return train_dataloader, val_dataloader, test_dataloader

def get_three_dataframes(image_folder, lstm = False, soft_train=False, include_disagreement=False):
    """
    Get images from the dataset directory, create pandas dataframes of image filepaths and ground truths.

    soft_train: if True (F1a intervention), the train split's C1/C2/C3 columns
    hold the raw mean annotator-agreement values in {0, 0.333, 0.667, 1.0}.
    Val and test always stay MV-rounded so that best-val-mAP selection during
    training and test-set scoring downstream use the same target as the
    locked-recipe baseline.

    include_disagreement: if True (F1b intervention), each frame's label is a
    6-vector [C1, C2, C3, D1, D2, D3] where D1/D2/D3 are the disagreement
    targets (1 = contested 2-1 split, 0 = unanimous). All three splits get
    the 6-vector format so the disagreement head can be trained on train,
    selected on val (MV mAP for best-val), and scored on test (AUC + AP).
    """
    # Specify directories for the splits
    train_dir = image_folder / 'train'
    val_dir  = image_folder / 'val'
    test_dir = image_folder / 'test'

    # Hardcoded to the MV annotation file. The original glob
    # `if 'json' and 'ds_coco' in x` short-circuits to `if 'ds_coco' in x`,
    # which also matches annotation_ds_coco_optimal.json and picks whichever
    # os.listdir returns first (alphabetical on NTFS, inode order on Linux —
    # different filesystems silently selected different ground truths).
    # The multihead experiment uses get_datasets_custom_json with an
    # explicit json_name; this single-head path needs the same explicitness.
    train_file = 'annotation_ds_coco.json'
    val_file = 'annotation_ds_coco.json'
    test_file = 'annotation_ds_coco.json'

    # Create dataframe with filepaths for individual images along with ground
    # truth labels. Train may be soft; val and test are always MV.
    train_dataframe = get_dataframe(train_dir / train_file, soft=soft_train)
    val_dataframe = get_dataframe(val_dir / val_file, soft=False)
    test_dataframe = get_dataframe(test_dir / test_file, soft=False)
    if lstm:
        # Add unlabelled images to the dataframe
        with open(image_folder / 'all' / 'annotation_coco.json', 'r') as file:
            all_images = json.load(file)  # Load JSON data
        all_image_names = [x['file_name'] for x in all_images['images']]

        # Add images to correct file lists according from which video they are
        train_images = [img for img in all_image_names if 1 <= int(img.split('_')[0]) <= 120]
        val_images = [img for img in all_image_names if 121 <= int(img.split('_')[0]) <= 161]
        test_images = [img for img in all_image_names if 162 <= int(img.split('_')[0]) <= 201]

        # Adding unlabelled images
        train_dataframe = add_unlabelled_imgs(train_images, train_dataframe)
        val_dataframe = add_unlabelled_imgs(val_images, val_dataframe)
        test_dataframe = add_unlabelled_imgs(test_images, test_dataframe)
   
        # Generate 5 frame sequences and update format to include paths to images
        train_dataframe = get_frame_sequence_dataframe(train_dataframe, train_dir,
                                                      include_disagreement=include_disagreement)
        val_dataframe = get_frame_sequence_dataframe(val_dataframe, val_dir,
                                                    include_disagreement=include_disagreement)
        test_dataframe = get_frame_sequence_dataframe(test_dataframe, test_dir,
                                                     include_disagreement=include_disagreement)
        return train_dataframe, val_dataframe, test_dataframe

    updated_train_dataframe = update_dataframe(train_dataframe, train_dir)
    updated_val_dataframe = update_dataframe(val_dataframe, val_dir)
    updated_test_dataframe = update_dataframe(test_dataframe, test_dir)
    return updated_train_dataframe, updated_val_dataframe, updated_test_dataframe

class Endoscapes_Dataset(Dataset):
    """
    Dataset creator only for backbone - SwinV2 training.
    """
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        image_info = self.image_dataframe.iloc[idx]
        image_path = image_info['path']
        label = torch.tensor(image_info['classification'])

        image = Image.open(image_path)
        
        if self.transforms:
            image = self.transforms(image)
            image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image))
      
        return image, label
    
class EndoscapesSwinCVS_Dataset(Dataset):
    """
    Dataset creator for SwinCVS - includes 5 frame sequences.
    """
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        sequence_info = self.image_dataframe.iloc[idx]
        image_f0_path = sequence_info['f0']
        image_f1_path = sequence_info['f1']
        image_f2_path = sequence_info['f2']
        image_f3_path = sequence_info['f3']
        image_f4_path = sequence_info['f4']
        paths = [image_f0_path, image_f1_path, image_f2_path, image_f3_path, image_f4_path]
        
        image_list = []
        if self.transforms:
            seed = random.randint(0, 2**32)
            for path in paths:
                image = Image.open(path)
                torch.manual_seed(seed)
                random.seed(seed)
                image = self.transforms(image)
                image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image))
                image_list.append(image)
        else:
            for path in paths:
                image = Image.open(path)
                image_list.append(image)
        
        images = torch.stack(image_list)
        label = torch.tensor(sequence_info['classification'])

        return images, label

def get_dataframe(json_path, soft=False):
    """
    Get dataframes of the dataset splits in columns:
    idx | vid | frame | C1 | C2 | C3 | D1 | D2 | D3

    soft: if False (default, baseline behaviour), C1/C2/C3 are MV-rounded
    integer labels in {0, 1}. If True (F1a intervention on the train split
    only), C1/C2/C3 are the raw mean annotator-agreement values in
    {0, 0.333, 0.667, 1.0}.

    D1/D2/D3 are the F1b disagreement targets — 1 iff the criterion's three
    annotators were NOT unanimous (2-1 split / contested), 0 if unanimous.
    Derived from ds independently of `soft`; always stored. Whether the
    dataset returns them is governed by `include_disagreement` in
    get_class / get_frame_sequence_dataframe / update_dataframe.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    vid = []
    frame = []
    C1 = []
    C2 = []
    C3 = []
    D1 = []
    D2 = []
    D3 = []

    for i in data['images']:
        # Extract data
        file_name = i['file_name']
        file_name = file_name.split('.')[0]
        file_name = file_name.split('_')
        vid_i = file_name[0]
        frame_i = file_name[1]
        if soft:
            C1_i = float(i['ds'][0])
            C2_i = float(i['ds'][1])
            C3_i = float(i['ds'][2])
        else:
            C1_i = round(i['ds'][0])
            C2_i = round(i['ds'][1])
            C3_i = round(i['ds'][2])
        # F1b disagreement: ds is the mean of three annotator binary votes,
        # so a strict (0, 1) value means a 2-1 split / contested frame.
        D1_i = int(0 < i['ds'][0] < 1)
        D2_i = int(0 < i['ds'][1] < 1)
        D3_i = int(0 < i['ds'][2] < 1)

        # Put in list
        vid.append(vid_i)
        frame.append(frame_i)
        C1.append(C1_i)
        C2.append(C2_i)
        C3.append(C3_i)
        D1.append(D1_i)
        D2.append(D2_i)
        D3.append(D3_i)

    data_dict = {'vid': vid,
                'frame': frame,
                'C1': C1,
                'C2': C2,
                'C3': C3,
                'D1': D1,
                'D2': D2,
                'D3': D3}
    data_dataframe = pd.DataFrame(data_dict)
    return data_dataframe

def get_frame_sequence_dataframe(dataframe, image_folder, include_disagreement=False):
    """
    For LSTM dataframe creator. Using dataframes updated with unlabelled images, get five frame sequences. The returned dataframe has columns:
    idx | f0 | f1 | f2 | f3 | f4 | classification
    idx - index of the sequence
    f0-4 - path to each image in the sequence
    classification - list of ground truth values. 3-vector [C1, C2, C3]
        for baseline / F1a, 6-vector [C1, C2, C3, D1, D2, D3] for F1b.
    """
    new_dataframe_rows = []
    # Iterate over each video so as not to create intravid sequences
    for video in dataframe['vid'].unique():
        temp_vid_dataframe = dataframe.loc[dataframe['vid'] == video]
        # Iterate over each datapoint in the dataframe
        for idx in range(len(temp_vid_dataframe)-5):
            # Extract 5 frame sequences
            five_seq_dataframe = temp_vid_dataframe.iloc[idx:idx+5]

            # Check if the last frame in the sequence is labelled
            if five_seq_dataframe.iloc[4]['C1'] != -1:
                # Update paths to images for all five frames
                paths = []
                for datapoint in five_seq_dataframe.iterrows():
                    paths.append(generate_path(datapoint[1], image_folder)) # CHANGE VAL_DIR!!!!
                # Get class of the last frame
                classification = get_class(five_seq_dataframe.iloc[4],
                                           include_disagreement=include_disagreement)
                # Put it in a new row of the dataframe
                new_row = { 'f0': paths[0], 'f1':  paths[1], 'f2': paths[2], 'f3': paths[3], 'f4': paths[4],
                            'classification': classification}
                new_dataframe_rows.append(new_row)

    updated_dataframe = pd.DataFrame(new_dataframe_rows)

    return updated_dataframe

def update_dataframe(dataframe, image_folder):
    """
    Function only for creation of dataframes when training backbone - SwinV2. It changes the structure of the dataframe from:
    idx | vid | frame | C1 | C2 | C3
    to:
    idx | path | classification
    where path is a path to a given image and classification is a list of ground truth values for C1-3 as, [C1, C2, C3] e.g. [0.0, 0.0, 1.0] 
    """
    dataframe['path'] = dataframe.apply(lambda row: generate_path(row, image_folder), axis=1)
    dataframe['classification'] = dataframe.apply(lambda row: get_class(row), axis=1)
    dataframe.drop(columns=['vid', 'frame', 'C1', 'C2', 'C3'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def add_unlabelled_imgs(list_of_selected_images, selected_dataframe):
    """
    Given existing splits dataframes, in correct order, append images that are unlabelled. Output dataframe has columns:
    idx | vid | frame | C1 | C2 | C3 | D1 | D2 | D3
    where C1-3 / D1-3 are unlabelled they have value -1. The sequence builder
    filters these out (it checks C1 != -1 on the last frame of the 5-frame
    window), so the disagreement target is never read on unlabelled frames.
    """
    rows = []
    for image in list_of_selected_images:
        contents = image.split('.')[0].split('_')
        frame_info = (contents[0], contents[1])
        rows.append({'vid': frame_info[0], 'frame': frame_info[1],
                     'C1': -1, 'C2': -1, 'C3': -1,
                     'D1': -1, 'D2': -1, 'D3': -1})

    df = pd.DataFrame(rows)

    combined_df = pd.merge(df, selected_dataframe, on=['vid', 'frame'], how='left', suffixes=('_new', '_lbld'))
    for col in ('C1', 'C2', 'C3', 'D1', 'D2', 'D3'):
        combined_df[col] = combined_df[col + '_lbld'].combine_first(combined_df[col + '_new'])

    # Drop the redundant columns from df1
    final_df = combined_df[['vid', 'frame', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']]

    final_df = final_df.sort_values(by=['vid', 'frame'])
    final_df = final_df.reset_index(drop=True)

    return final_df

def generate_path(row, image_folder):
    vid = row['vid']
    frame = row['frame']
    filename = str(vid) + '_' + str(frame) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def get_class(row, include_disagreement=False):
    """
    Return the per-frame label list.

    Default (include_disagreement=False): 3-vector [C1, C2, C3] — baseline / F1a.
    F1b (include_disagreement=True):     6-vector [C1, C2, C3, D1, D2, D3].
    The training loop slices [:3] for the MV-head loss and [3:] for the
    disagreement-head loss.
    """
    classification = [float(row['C1']), float(row['C2']), float(row['C3'])]
    if include_disagreement:
        classification += [float(row['D1']), float(row['D2']), float(row['D3'])]
    return classification

def get_endoscapes_mean_std(config):
    mean = config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN
    std = config.TRAIN.TRANSFORMS.ENDOSCAPES_STD

    # Change BGR to RGB
    mean = mean[::-1]
    std = std[::-1]

    return mean, std

def get_transform_sequence(config):
    mean, std = get_endoscapes_mean_std(config)
    transform_sequence = transforms.Compose([   transforms.CenterCrop(config.TRAIN.TRANSFORMS.CENTER_CROP),
                                                transforms.Resize((384, 384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=torch.tensor(mean),
                                                    std=torch.tensor(std))])
    return transform_sequence