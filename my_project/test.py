from data import SampleGenerator
from config import Config
from models_engine import ModelEngine
from model.xDeepFM import xDeepFM
import fire


def model_predict(path):
    Config["normal_config"]["pretrain"] = True
    Config["normal_config"]["pretrain_model_dir"] = path
    engine = ModelEngine(config=Config, model=xDeepFM)
    sample_generator = SampleGenerator()
    
    print()
    print("------------start evaluate-------------")
    evaluate_loader = sample_generator.instance_a_loader(t="val")
    auc = engine.evaluate(evaluate_loader, epoch_id=10)
    print("------------finish evaluate------------")
    print("maybe you will modify this code and remove evaluation coding if pretrain is correct*************")

    print()
    print("------------start testing--------------")
    test_loader = sample_generator.instance_a_loader(t="test")
    engine.predict(test_loader)
    print("------------finish testing")


fire.Fire(model_predict)