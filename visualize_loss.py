import matplotlib.pyplot as plt
import pickle
import sys

# this script is used to turn the loss_history file into a diagram
if len(sys.argv) < 2:
    run_number = 1
else:
    run_number = sys.argv[1]

# create a figure and axis
with open("run"+str(run_number)+"/loss_history.p", "rb") as f:
    loss_array = pickle.load(f)
loss_array = [x[0] for x in loss_array]
print(loss_array)
iteration = range(1, len(loss_array)+1)
plt.plot(iteration, loss_array, color='g')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss-Progression')
plt.show()
