import torch
import numpy as np
import task_generator as tg
import miniimagenet_train_few_shot_experiment as ot
import os
import random
import scipy as sp
import scipy.stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

FEATURE_DIM = 64  # args.feature_dim
RELATION_DIM = 8  # args.relation_dim
CLASS_NUM = 5  # args.class_num
SAMPLE_NUM_PER_CLASS = 5  # args.sample_num_per_class
BATCH_NUM_PER_CLASS = 15  # args.batch_num_per_class
EPISODE = 1000000  # args.episode
TEST_EPISODE = 600  # args.test_episode
LEARNING_RATE = 0.001  # args.learning_rate
# GPU = # args.gpu
HIDDEN_UNIT = 10  # args.hidden_unit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # * Step 1: init data folders
    print("init data folders")
    
    # * init character folders for dataset construction
    metatrain_folders, metatest_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = ot.CNNEncoder().to(device)
    # RFT = RandomForestClassifier(n_estimators=100, random_state=1, warm_start=True)
    relation_network = ot.RelationNetwork(FEATURE_DIM, RELATION_DIM).to(device)
    
    feature_encoder.eval()
    relation_network.eval()    
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    '''     
    if os.path.exists(str("./models/miniimagenet_random_forest_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        RFT = pickle.load(open(str("./models/miniimagenet_random_forest_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"), 'rb'))
        print("load random forest success")
    '''   
    if os.path.exists(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")

    total_accuracy = 0.0
    
    for episode in range(10):
        # * test
        print("Testing...")
        
        accuracies = []
        
        for i in range(100):
            total_rewards = 0
            # degrees = random.choice([0, 90, 180, 270])
            task = tg.MiniImagenetTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)
            test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=5, split="test", shuffle=False)
            
            sample_images, sample_labels = next(iter(sample_dataloader))
            sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
            
            for test_images, test_labels in test_dataloader:
                batch_size = test_labels.shape[0]
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                # * Calculate features
                sample_features = feature_encoder(sample_images)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                test_features = feature_encoder(test_images)
                
                sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                
                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                
                _, predict_labels = torch.max(relations.data, 1)
                
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                
            accuracy = total_rewards / (1.0 * CLASS_NUM * 15)
            accuracies.append(accuracy)
            
        test_accuracy, h = mean_confidence_interval(accuracies)
        print(f'test accuracy : {test_accuracy:.4f}, h : {h:.4f}')
        total_accuracy += test_accuracy
        
    print(f"average accuracy : {total_accuracy/10 :.4f}")
    
    
if __name__ == "__main__":
    main()
