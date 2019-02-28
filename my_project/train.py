from model.mlp import MLPEngine
from data import SampleGenerator
from config import Config


engine = MLPEngine(Config)
sample_generator = SampleGenerator()

global_step = 0
for epoch in range(Config['num_epoch']):
    print('Epoch {} starts!'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_loader(t="train")
    engine.train_an_epoch(train_loader, epoch_id=global_step)
    
    evaluate_loader = sample_generator.instance_a_loader(t="val")
    test_score, auc = engine.evaluate(evaluate_loader, epoch_id=epoch)
    engine.save(epoch, auc=auc)

# close hd5 file
