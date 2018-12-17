# By Carlos Leon
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# global
IMAGE_OUTPUT = "./RESULTS/IMAGES/"

# function to select k
def select_k(data, ids, debug, run=0, show=True):
	"""
		Function to select best k for KNN
	"""
	# set range
	low = 1
	high = 51

	# Neighbors
	neighbors = [x for x in range(low, high) if x % 2 !=0]

	# Metrics
	metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]

	# Create empty list that will hold cv scores
	cv_scores = []

	# Perform 10-fold cross validation on training set for odd values of k:
	for m in metrics:
		# Create an empty list for each metric score
		k_scores = []

		for k in neighbors:
			# Instantiate KNN with k value and m metric
			knn = KNeighborsClassifier(n_neighbors=k, metric=m)

			# Instantiate 10 fold cross-validation
			kfold = model_selection.KFold(n_splits=10)

			# Acquire scores
			scores = model_selection.cross_val_score(estimator=knn, x=data, y=ids, cv=kfold, scoring='accuracy', error_score='raise')
			k_scores.append(scores.mean()*100)

		# Save the metric scores
		cv_scores.append(k_scores)
 
	# Note optimal K and metric
	highest_metrics = []
	for i in range(len(cv_scores)):
		highest_metrics.append(max(cv_scores[i]))
	idx = highest_metrics.index(max(highest_metrics))
	optimal_metric = metrics[idx]
	optimal_k = neighbors[cv_scores[idx].index(max(cv_scores[idx]))]
	debug += "The optimal number of neighbors is %d using %s with %0.1f%%\n" % (optimal_k, optimal_metric, cv_scores[idx][optimal_k])
	
	# Plot the results
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Train Accuracy')
	plt.title("K Accuracy")
	index = 0
	for m in metrics:
		plt.plot(neighbors, cv_scores[index], label=m)

	# save image
	plt.savefig(IMAGE_OUTPUT + 'k_accuracy_' + str(run) + '.png', bbox_inches='tight')

	# show
	if show:
		plt.show()

	# return
	return optimal_k, optimal_metric, debug
