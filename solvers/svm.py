from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    # import your reusable functions here
    from sklearn.svm import SVC


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'
    # and are set to one value of the list.
    parameters = {
        'kernel': ['linear', 'poly', 'sigmoid'],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py. Here `scikit-learn` is already present
    # so no need to add it again.
    requirements = []

    # Force the solver to run only once if you don't want to record training steps
    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.y_train = X_train, y_train
        self.clf = SVC(kernel=self.kernel)

    def run(self, _):
        # This is the method that is called to fit the model.
        # The input param is only defined if you change the sampling strategy
        # to value different than "run_once".
        # See https://benchopt.github.io/performance_curves.html
        self.clf.fit(self.X_train, self.y_train)

    def get_result(self):
        # Returns the model after fitting.
        # The output of this function is a dictionary whose keys define the
        # keyword arguments for `Objective.evaluate_result`.
        # This defines the benchmark's API for solvers' results.
        # It is customizable for each benchmark.
        return dict(model=self.clf)
