import os
import tensorflow as tf
from utils.utils import get_dataset_path_list
from common.model_args import init_model_args
from models.model import RecommendModelHandler


def main():

  # basic logging setup for tf
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  tf.logging.set_verbosity(tf.logging.INFO)

  # init necessary args
  # args = init_model_args()

  #train_dataset_path_list = get_dataset_path_list(train_dataset_path, sub_str="track2_train_time.txt")
  train_dataset_path_list = "../../../data/track2/final_track2_train.txt"

  val_dataset_path_list = "../../../data/track2/final_track2_train.txt"
  print ("training path list: {}".format(train_dataset_path_list))
  print ("training path list: {}".format(val_dataset_path_list))

  save_model_dir = "../../../data/"
  print ("saving model in ... {}".format(save_model_dir))

  optimizer = "adam"
  learning_rate  = 0.0005
  print ("we use {} as optimizer".format(optimizer))
  print ("learning rate is set as  {}".format(learning_rate))

  batch_size = 128
  embedding_size = 8
  num_epochs = 1
  print ("batch size: {}".format(batch_size))
  print ("embedding size: {}".format(embedding_size))

  task = "finish"
  track = 2
  print ("track: {}, task: {}".format(track, task))


  model = RecommendModelHandler(
      train_dataset_path=train_dataset_path_list,
      val_dataset_path=val_dataset_path_list,
      save_model_dir=save_model_dir,
      num_epochs=num_epochs,
      optimizer=optimizer,
      batch_size= batch_size,
      embedding_size=embedding_size,
      task=task,
      track=track,
      learning_rate=learning_rate)

  model.train()

if __name__ == '__main__':
  main()
