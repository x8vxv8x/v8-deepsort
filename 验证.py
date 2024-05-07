from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('111.pt')
    metrics = model.val(data=r'E:\datasets\archive2\drone_dataset\data.yaml')




