from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    prec = float(input("Input precision threshold: "))
    path = input("Input experiment name: ")
    mn = input("Input model name: ")
    src = input("Input image or video sources: ")
    model = YOLO(mn)
    results = model.predict(
        source = src,
        name=path,
        save=True,
        show=True,
        half=True,
        stream=True,
        conf = prec
    )
    for result in results:
        pass
    