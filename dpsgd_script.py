import subprocess
import mnist

def call_script1():
    # Define the arguments to pass to script1.py
    script_name = "mnist.py"
    #--backbone_name "shaders21k_grey" --delta 10e-5 --epochs 10 --data-root "/storage3/michel/data/"
    
    prefix_args = ["--backbone_name"]
    backbone_names= [None, "shaders21k_grey", "stylegan", "dead_leaves"]
    suffix_args = ["--delta", "10e-5", "--epochs", "1", "--data-root", "/storage3/michel/data/"] 
    
    print("test")
    # Call script1.py using subprocess
    all_accuracies = []
    eps=[]
    for backbone_name in backbone_names:
        result, epsilon = mnist.main(backbone_name)
        all_accuracies.append(result)
        eps.append(epsilon)
    
    print(all_accuracies)
    print(eps)
  

if __name__ == "__main__":
    call_script1()