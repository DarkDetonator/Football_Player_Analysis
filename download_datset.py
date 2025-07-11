from roboflow import Roboflow
rf = Roboflow(api_key="nXApD4Y4pSVZo90ArsIv")
project = rf.workspace("kim-6sqps").project("football-g2bqg")
version = project.version(5)
dataset = version.download("yolov11")