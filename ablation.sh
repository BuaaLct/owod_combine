python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MOMENTUM 0.4 OUTPUT_DIR "./output/momentum_0_4"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MOMENTUM 0.5 OUTPUT_DIR "./output/momentum_0_5"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MOMENTUM 0.6 OUTPUT_DIR "./output/momentum_0_6"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52132' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.ITEMS_PER_CLASS 10 OUTPUT_DIR "./output/items_10"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.ITEMS_PER_CLASS 30 OUTPUT_DIR "./output/items_30"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52134' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.ITEMS_PER_CLASS 50 OUTPUT_DIR "./output/items_50"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52131' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.ITEMS_PER_CLASS 5 OUTPUT_DIR "./output/items_5"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52136' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MARGIN 5.0 OUTPUT_DIR "./output/margin_5"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52137' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MARGIN 15.0 OUTPUT_DIR "./output/margin_15"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52135' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MARGIN 1.0 OUTPUT_DIR "./output/margin_1"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52138' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MARGIN 20.0 OUTPUT_DIR "./output/margin_20"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52128' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MOMENTUM 0.7 OUTPUT_DIR "./output/momentum_0_7"
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.CLUSTERING.MOMENTUM 0.8 OUTPUT_DIR "./output/momentum_0_8"