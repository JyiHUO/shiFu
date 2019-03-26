from data import SampleGenerator
from config import Config
from models_engine import ModelEngine
from model.xDeepFM import xDeepFM
from model.nffm import NFFM
import fire


def model_predict(path_finish, path_like):
    Config["normal_config"]["pretrain"] = True
    Config["normal_config"]["pretrain_model_dir"] = path_finish
    engine_finish = ModelEngine(config=Config, model=NFFM)

    sample_generator = SampleGenerator()

    print()
    print("------------start testing finish--------------")
    test_loader = sample_generator.instance_a_loader(t="test")
    df_finish = engine_finish.predict(test_loader)
    print("------------finish testing -------------------")
    print("------------start testing like ----------------")
    Config["normal_config"]["pretrain_model_dir"] = path_like
    engine_like = ModelEngine(config=Config, model=NFFM)
    df_like = engine_like.predict(test_loader)

    df_finish["like_probability"] = df_like["pred_probability"]
    df_finish.columns = ["uid", "item_id","finish_probability", "like_probability"]
    df_finish.to_csv(Config["normal_config"]["predict_file"] + Config["normal_config"]["model_name"],
                     index=None, float_format="%.6f")


fire.Fire(model_predict)