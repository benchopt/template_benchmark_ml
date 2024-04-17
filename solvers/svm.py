from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    # import your reusable functions here
    from sklearn import svm


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'kernel': ['linear', 'poly', 'sigmoid'],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ['sklearn']

    # Force solver to run only once if you don't want to record training steps
    sampling_strategy = "run_one"

    def set_objective(self, X_train, y_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.y_train = X_train, y_train
        self.clf = svm.SVC(kernel=self.kernel)

    def run(self, n_iter):
        # This is the function that is called to fit the model.
        # The param n_iter is defined if you change the sample strategy to other value than "run_once"
        # https://benchopt.github.io/performance_curves.html
        self.clf.fit(self.X_train, self.y_train)

    def get_result(self):
        # Returns the model after fitting.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.clf)
