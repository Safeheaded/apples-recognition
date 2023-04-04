# from IPython import display
# display.clear_output()

import ultralytics
# ultralytics.checks()

from ultralytics import YOLO

# from IPython.display import display, Image

from roboflow import Roboflow
# print(rf.workspace())

import yaml
from dotenv import load_dotenv
import os



if __name__ == '__main__':
    load_dotenv()
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("knpm").project("apples-detection-vqjng")
    dataset = project.version(1).download("yolov8")
    documents = None
    with open(fr"{dataset.location}/data.yaml", "r") as f:
        documents = yaml.load(f, Loader=yaml.FullLoader)
        documents['train'] = f"{dataset.location}/{documents['train']}"
        documents['val'] = f"{dataset.location}/{documents['val']}"
        documents['test'] = f"{dataset.location}/{documents['test']}"
        with open(fr"{dataset.location}/data.yaml", "w") as f:
            yaml.dump(documents, f)
            f.close()
    model = YOLO("yolov8n.pt")
    model.to('cuda')
    model.train(data="./apples-detection-1/data.yaml", epochs=5)
    model.export(format='onnx')
