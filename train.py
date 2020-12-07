from utils import TargetTransform, read_and_preprocess



if __name__ == "__main__":
    targetTransform = TargetTransform(transform_power=2)
    train, test = read_and_preprocess(targetTransform)

    print(train.shape)
    print(test.shape)
