import os
import datetime
import json
import zipfile
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi

project_dir = os.getcwd()

dataset_config_path = os.path.join(
    project_dir, 'configs', 'datasets_config.json')
with open (dataset_config_path) as json_config:
    dataset_config_options = json.load(json_config)

home_dir = os.path.expanduser('~')
# https://towardsdatascience.com/
# downloading-datasets-from-kaggle-for-your-ml-project-b9120d405ea4
# Assuming kaggle config file is in ~/.kaggle
with open (os.path.join(home_dir, '.kaggle', 'kaggle.json')) as json_config:
    kaggle_config = json.load(json_config)
os.environ['KAGGLE_USERNAME'] = kaggle_config['username']
os.environ['KAGGLE_KEY'] = kaggle_config['key']

def main():
    for dataset_key, dataset_config in dataset_config_options.items():
        print(f'Downloading {dataset_key}')
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        io_dir = os.path.join(project_dir, 'data', dataset_key)
        raw_dir = os.path.join(io_dir, 'raw')
        processed_dir = os.path.join(io_dir, 'processed')
        for d in [io_dir, raw_dir, processed_dir]:
            if not os.path.exists(d):
                os.mkdir(d)
        print(f'Files in dataset directory: {os.listdir(io_dir)}')
        # This could be generalized to include logic to check for the
        # kaggle_identifier key and use the url with web scraping if
        # it doesn't exist or some other appropriate method
        kaggle_identifier = dataset_config['kaggle_identifier']
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        dataset_list_files = (kaggle_api
                              .dataset_list_files(kaggle_identifier)
                              .files)
        print(f'Kaggle files: {dataset_list_files}')
        # TODO: generalize for more than 1 file
        unzipped_raw_data_filename = str(dataset_list_files[0])
        raw_data_filename, raw_data_ext = os.path.splitext(
            unzipped_raw_data_filename)
        raw_data_filename = f'{raw_data_filename}_{current_date}'
        raw_data_filename = f'{raw_data_filename}{raw_data_ext}'
        raw_data_path = os.path.join(raw_dir, raw_data_filename)
        kaggle_api.dataset_download_files(
            kaggle_identifier, path=raw_dir)
        print(f'Files in raw data directory: {os.listdir(raw_dir)}')

        zipped_raw_data_path = os.path.join(raw_dir, f'{dataset_key}.zip')
        with zipfile.ZipFile(zipped_raw_data_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zipped_raw_data_path)
        # zipfile extracts the files into the current working directory
        unzipped_raw_data_path = os.path.join(
            os.getcwd(), unzipped_raw_data_filename)
        print(f'File in the current directory: {os.listdir(os.getcwd())}')
        if (os.path.exists(unzipped_raw_data_path)
            and not os.path.exists(raw_data_path)):
            os.rename(unzipped_raw_data_path, raw_data_path)
        elif os.path.exists(unzipped_raw_data_path):
            os.remove(unzipped_raw_data_path)
        print(f'Files in raw data directory: {os.listdir(raw_dir)}')

        md5_checksum = ''
        with open(raw_data_path, 'rb') as file_to_check:
            file_data = file_to_check.read()
            md5_checksum = hashlib.md5(file_data).hexdigest()
        if md5_checksum:
            md5_checksum_path = os.path.join(raw_dir, 'md5checksum.txt')
            if os.path.exists(md5_checksum_path):
                with open(md5_checksum_path, 'a') as checksum_file:
                    checksum_file.write(f'{md5_checksum},{current_date}\n')
            else:
                with open(md5_checksum_path, 'w') as checksum_file:
                    checksum_file.write('md5,date\n')
                    checksum_file.write(f'{md5_checksum},{current_date}\n')
        print(f'MD5 checksum: {md5_checksum}')

        print(f'Files in raw data directory: {os.listdir(raw_dir)}')
        # Update the datasets_config file with any new key:value pairs
        # Set to latest downloaded filename
        dataset_config['raw_data_filename'] = raw_data_filename
        dataset_config_options[dataset_key] = dataset_config
        print('------------------------------------------------------------\n')
    with open(dataset_config_path, 'w') as json_out:
        json.dump(dataset_config_options, json_out, indent=4)
    print('datasets_config.json has been updated')

if __name__ == '__main__':
    main()