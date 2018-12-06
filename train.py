import matplotlib.pyplot as plt
from DataSet import Dataset
import Test_train
from InstrumentNet import InstrumentNet
import torch
import torch.optim as optim
import pt_util
from torchvision import transforms
# Play around with these constants, you may find a better setting.
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
USE_CUDA = True
PRINT_INTERVAL = 100
WEIGHT_DECAY = 0.0005

####INPUTS To Change####
trainPath = 'nsynth-valid/audio/'
testPath = 'nsynth-test/audio/'
LOG_PATH = 'logs/log.pkl'


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])
data_train = Dataset(trainPath, transform=train_transforms)
data_test = Dataset(testPath, transform=test_transforms)

###################################################

loss=[]
Acc=[]
Train_loss=[]
# Now the actual training code
use_cuda = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
import multiprocessing
print('num cpus:', multiprocessing.cpu_count())

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}



train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                          shuffle=False, **kwargs)

model = InstrumentNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
start_epoch = model.load_last_model('checkpoints/')

# You may want to define another default for your log data depending on how you save it.
log_data = pt_util.read_log(LOG_PATH, [])

correct_images, correct_val, error_images, predicted_val, gt_val,accuracy,Testloss = Test_train.test(model, device, test_loader, True)
correct_images = pt_util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
error_images = pt_util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
#pt_util.show_images(correct_images, ['correct: %s' % class_names[aa] for aa in correct_val])
#pt_util.show_images(error_images, ['pred: %s, actual: %s' % (class_names[aa], class_names[bb]) for aa, bb in zip(predicted_val, gt_val)])

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        loss.append(Testloss)
        Acc.append(accuracy)
        trainloss=Test_train.train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        Train_loss.append(trainloss)
        correct_images, correct_val, error_images, predicted_val, gt_val,accuracy,Testloss = Test_train.test(model, device, test_loader, True)
        model.save_best_model(accuracy,DATA_PATH + 'checkpoints/%03d.pt' % epoch, 0)
        # TODO define other things to do at the end of each loop like logging and saving the best model.


except KeyboardInterrupt as ke:
    print('Interrupted')
except:
    import traceback
    traceback.print_exc()
finally:
    # Always save the most recent model, but don't delete any existing ones.
    model.save_model(DATA_PATH + 'checkpoints/%03d.pt' % epoch, 0)

    # Show some current correct/incorrect images.
    correct_images = pt_util.to_scaled_uint8(correct_images.transpose(0, 2, 3, 1))
    error_images = pt_util.to_scaled_uint8(error_images.transpose(0, 2, 3, 1))
    pt_util.show_images(correct_images, ['correct: %s' % class_names[aa] for aa in correct_val])
    pt_util.show_images(error_images, ['pred: %s, actual: %s' % (class_names[aa], class_names[bb]) for aa, bb in zip(predicted_val, gt_val)])
