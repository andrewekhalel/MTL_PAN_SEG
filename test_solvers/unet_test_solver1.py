from subprocess import call

#NOTE: mean_list is calculated from imagenet dataset, which contains millions of images

#configuration of the network for the test phase
call(["python3", "main.py", 
      "--is_training=False",
      "--train_dir=sample/",
      "--valid_dir=sample/", 
      "--patch_size=1024", 
      "--padding=0",
      "--seg_enable=True",
      "--pansharp_enable=True",
      "--num_of_classes=1",
      "--snap_dir=weights/exp10_trees_both_alpha0.2"])

