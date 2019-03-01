from data import SampleGenerator
from config import Config
from models_engine import ModelEngine
from model.xDeepFM import xDeepFM
from model.mlp import MLP

engine = ModelEngine(config=Config, model=xDeepFM)
sample_generator = SampleGenerator()

for epoch in range(Config["training_config"]['num_epoch']):
    print('Epoch {} starts!'.format(epoch))
    print('-' * 80)
    evaluate_loader = sample_generator.instance_a_loader(t="val")
    auc = engine.evaluate(evaluate_loader, epoch_id=epoch)
    engine.save(epoch, auc=auc)

    train_loader = sample_generator.instance_a_loader(t="train")
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    


# close hd5 file
