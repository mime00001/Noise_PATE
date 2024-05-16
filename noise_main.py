import models.resnet10
import models.resnet12
import models.resnet9
import models.mnistresnet
from utils import teachers
import models

a = models.mnistresnet.Target_Net(1, 1)


teachers.util_train_teachers("MNIST", n_epochs=50, nb_teachers=200, lr=0.001, weight_decay=0)

#kd lr=0.001, decay =0
#CAPC epochs=500, nb_teachers = 50, lr=0.01, decay = 1e-5

#for mnist epochs = 35, lr = 0.001, decay = 0