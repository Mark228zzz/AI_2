import kagglehub

def download_data():
    # Download dataset into ~/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1
    path = kagglehub.dataset_download('uciml/sms-spam-collection-dataset')

    print(f'Dataset downloaded to: {path}')

if __name__ == '__main__':
    download_data()
