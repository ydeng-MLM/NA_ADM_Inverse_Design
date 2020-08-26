"""
This is the function that generates the gaussian mixture for artificial model
The simulated cluster would be similar to the artifical data set from the INN paper
"""
# Import libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Define the free parameters
dimension = 2
num_cluster = 8
num_points_per_cluster = 1000
num_class = 4
cluster_distance_to_center = 10
in_class_variance = 1


def plotData(data_x, data_y, save_dir='generated_gaussian_scatter.png', eval_mode=False):
    """
    Plot the scatter plot of the data to show the overview of the data points
    :param data_x: The 2 dimension x values of the data points
    :param data_y: The class of the data points
    :param eval_mode: Bool, is this evaluation mode
    :param save_dir: The save name of the plot
    :return: None
    """
    f = plt.figure(figsize=[15,15])
    if 'Tandem' in save_dir:
        plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, s=100,cmap='brg')
    else: 
        if eval_mode:
            data_x *= cluster_distance_to_center + in_class_variance
        plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, s=5,cmap='brg')
    plt.title(save_dir, fontsize=15)
    plt.xlabel('x1')
    plt.xlabel('x2')
    f.savefig(save_dir)


def determine_class_from_x(Xpred, data_dir='../Simulated_DataSets/Gaussian_Mixture/'):
    """
    Determine the class label from Xpred
    :param Xpred: [N, 2] The xpred to be simulated
    :return: Ypred: [N, 1] The label of those Xpred points
    """
    labels = np.loadtxt(data_dir + 'class_labels.csv', delimiter=',')
    centers = np.loadtxt(data_dir + 'data_centers.csv', delimiter=',')
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
    distance, indices = nbrs.kneighbors(Xpred)
    Ypred = np.zeros([len(Xpred), 1])
    # print(indices)
    for i in range(len(Xpred)):
        Ypred[i] = labels[indices[i]]
    return Ypred



if __name__ == '__main__':
    centers = np.zeros([num_cluster, dimension])            # initialize the center positions
    for i in range(num_cluster):                            # the centers are at the rim of a circle with equal angle
        centers[i, 0] = np.cos(2 * np.pi / num_cluster * i) * cluster_distance_to_center
        centers[i, 1] = np.sin(2 * np.pi / num_cluster * i) * cluster_distance_to_center
    print("centers", centers)
    # Initialize the data points for x and y
    data_x = np.zeros([num_cluster * num_points_per_cluster, dimension])
    data_y = np.zeros(num_cluster * num_points_per_cluster)
    # allocate the class labels
    class_for_cluster = np.random.uniform(low=0, high=num_class, size=num_cluster).astype('int')
    print("class for cluster", class_for_cluster)
    # Loop through the points and assign the cluster x and y values
    for i in range(len(data_x[:, 0])):
        i_class = i // num_points_per_cluster
        data_y[i] = class_for_cluster[i_class]             # Assign y to be 0,0,0....,1,1,1...,2,2,2.... e.g.
        data_x[i, 0] = np.random.normal(centers[i_class, 0], in_class_variance)
        data_x[i, 1] = np.random.normal(centers[i_class, 1], in_class_variance)
    #print("data_y", data_y)
    plotData(data_x, data_y)
    # Save the data into format
    np.savetxt('data_centers.csv', centers, delimiter=',')
    np.savetxt('class_labels.csv', class_for_cluster, delimiter=',')
    #np.savetxt('data_x.csv', data_x, delimiter=',')
    #np.savetxt('data_y.csv', data_y, delimiter=',')
