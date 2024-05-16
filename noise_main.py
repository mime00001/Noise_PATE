import models.resnet10
import models.resnet12
import models.resnet9
from utils import teachers
import models


teachers.util_train_teachers("CIFAR10", n_epochs=50, nb_teachers=50, lr=0.001, weight_decay=0)

#kd lr=0.001, decay =0
#CAPC epochs=500, nb_teachers = 50, lr=0.01, decay = 1e-5

