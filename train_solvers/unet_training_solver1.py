from subprocess import call

#NOTE: mean_list is calculated from imagenet dataset, which contains millions of images

#configuration of the network for the training phase

call(["python3", "main.py",
      "--is_training=True",
      "--fine_tuning=True",
      "--train_dir=../../datasets/ISPRS/vaihingen/simulated/tiles/train/", 
      "--valid_dir=../../datasets/ISPRS/vaihingen/simulated/tiles/test/",
      "--num_of_classes=1",
      "--batch_size=16",
      "--learning_rate=1e-3",
      "--num_of_epochs=60",
      "--num_of_iterations=5000",
      "--decay_epoch=10",
      "--decay_rate=0.1",
      "--seg_enable=True",
      "--pansharp_enable=True",
      "--snap_dir=snaps/exp1",
      "--snap_freq=5",
      "--log_dir=logs/exp1"])

