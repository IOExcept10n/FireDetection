from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    prec = float(input("Input precision threshold: "))
    path = input("Input experiment name: ")
    mn = input("Input model name: ")
    model = YOLO(mn)
    results = model.predict(
        source = "./Tests/*.mp4",
        name=path,
        save=True,
        show=True,
        half=True,
        stream=True,
        conf = prec
    )
    for result in results:
        pass
    