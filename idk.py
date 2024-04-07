from ultralytics import YOLO
import torch
import os

def Train(model: YOLO, epoch = 100):
    #currentEpochs = len(model.trainer.epochs)
    model.train(
        data="Training/data.yaml", 
        epochs=epoch,
        patience=epoch//2,
        batch=-1,
        exist_ok=True,
        imgsz=640, 
        save_period=25, 
        cache=True, 
        device=0, 
        project="Checkpoints", 
        name="IHopeThisIsLastOneM",
        #resume=True,
        plots=True)
    model.export(format="onnx")
    model.save("./_last.pt")

def Predict(model: YOLO, prec: float, path: str):
    PredictPart(model, "*.jpg", prec, path)
    PredictPart(model, "*.png", prec, path)
    PredictPart(model, "*.mp4", prec, path)

def PredictPart(model:YOLO, filter:str, prec:float, path: str):
    try:
        model.predict(
            source = "./Tests/"+filter,
            name=path,
            save=True,
            show=True,
            half=True,
            conf = prec
        )
    except:
        print("Unexpected error while prediction")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    mn = input("Input model filename: ")
    model = YOLO(mn)
    epochs = int(input("Input epochs: "))
    if (epochs > 0):
        Train(model, epochs)
    print("==== [Prediction] ====")
    prec = float(input("Input precision threshold: "))
    path = input("Input results directory path: ")
    Predict(model, prec, path)
    print("==== [Export] ====")
    model.export(format="onnx")
