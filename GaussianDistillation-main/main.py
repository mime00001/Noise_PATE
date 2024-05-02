import utils.train_teacher
import distill_gaussian

#utils.train_teacher.util_train_teacher("CIFAR10", 50)

distill_gaussian.experiment_distil_gaussian("CIFAR10", 75, 75, compare=True, student_model="resnet18")