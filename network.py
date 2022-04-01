import argparse
from data.indexes import SUPPORTED_DATASETS
from procedures import evaluate, test_indexes
# Modes:
#   Train - training the network
#   Eval  - evaluate given image pair and return resulting disparity
#   Test  - test various functions

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode",type=str,choices=["train","evaluate","test"],metavar="mode")

    datasets_group = parser.add_argument_group("Dataset", "Arguments used for dataset handling")
    datasets_group.add_argument("--dataset", dest="name",choices=SUPPORTED_DATASETS,action="store",help="Dataset used for training")
    datasets_group.add_argument("--root","--image-root",dest="root_images",action="store",type=str,help="Root folder with dataset")
    datasets_group.add_argument("--disparity-root",dest="root_disparity",action="store",type=str,help="Folder with disparity for sceneflow sets")
    datasets_group.add_argument("--split",dest="split",action="store",type=float,default=0.8,help="Percentage of dataset used for training")
    datasets_group.add_argument("--occlussion",dest="occlussion",action="store_true",help="If occludded disparity should be used for Kitti sets")
    datasets_group.add_argument("--colored",dest="colored",action="store_true",help="If colored input should be used for Kitti2012 set")
    datasets_group.add_argument("--no-webp",dest="webp",action="store_false",help="If dataset is not using webp in sceneflow sets")
    datasets_group.add_argument("--disparity-side",dest="disparity_side",type=str,default="left",help="Which side should be used for ground truth in sceneflow sets")

    model_group = parser.add_argument_group("Model","Arguments for model tweaking and loading")
    model_group.add_argument("--max-disp",dest="max_disp",action="store",type=int,help="What's the maximal disparity used in the model")
    model_group.add_argument("--cpu",dest="cpu",action="store_true",help="If model should use CPU")
    model_group.add_argument("--save",dest="save_file",action="store",help="Path to file in which data should be saved")
    model_group.add_argument("--load",dest="load_file",action="store",help="Path to file which should be loaded")

    train_group = parser.add_argument_group("Training","Arguments used for training mode")
    train_group.add_argument("--epochs",dest="epochs",type=int,help="How many epochs of training will be done")
    train_group.add_argument("--batch-size",dest="batch_size",type=int,help="How many images should be passed at once through the network")
    train_group.add_argument("--learning-rate",dest="learning_rate",type=float,help="Learning rate for the optimizer to start with")

    eval_group = parser.add_argument_group("Evaluation","Arguments used for evaluation mode")
    eval_group.add_argument("--left-image",dest="left_image",type=str, help = "Path to left image for evaluation")
    eval_group.add_argument("--right-image",dest="right_image",type=str, help="Path to right image for evaluation")
    eval_group.add_argument("--disparity-image",dest="disparity_image",type=str, help="Path to disparity image for evaluation")
    eval_group.add_argument("--result-image",dest="result_image",type=str, help="Path under which the result will be saved")
    eval_group.add_argument("--timed",dest="timed",action="store_true",help="If evaluation should be timed (takes longer than without)")

    test_group = parser.add_argument_group("Test","Arguments used for test mode")
    test_group.add_argument("--indexes",dest="test_indexes",action="store_true",help="If indexes should be tested")
    test_group.add_argument("--loading", dest="test_loading",action="store_true",help="If loading of data should be tested")
    # Tests for indexing each set
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["root"] = args_dict.get("root_images")
    # print(args)
    # for x in datasets_group._group_actions:
    #     print(x.dest,args.__dict__[x.dest])
    # return datasets_group,args
    if args.mode == "test":
        return setupTest(args)
    elif args.mode == "train":
        return setupTraining(args)
    elif args.mode == "evaluate":
        return setupEvaluation(args)


def setupTest(args: argparse.Namespace):
    kwargs = vars(args)
    # print(kwargs)
    if args.test_indexes:
        return test_indexes(kwargs)
    return


def setupEvaluation(args):
    kwargs = vars(args)
    if not kwargs.get("left_image"):
        print("left image path not defined, please use --left-image flag")
        return
    if not kwargs.get("right_image"):
        print("right image path not defined, please use --right-image flag")
        return
    if not kwargs.get("result_image"):
        print("result image path not defined, please use --result-image flag")
        return
    return evaluate(**kwargs)
    raise NotImplementedError("Evaluation mode is not implemented yet")


def setupTraining(args):
    raise NotImplementedError("Training mode is not implemented yet")

if __name__ == "__main__":
    main()