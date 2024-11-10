Data in the form of a table
Features in the form of a matrix
Label or target array


General principles
Consistency. All objects (basic or composite) share a consistent interface composed of a limited set of methods. This interface is documented in a consistent manner for all objects.

Inspection. Constructor parameters and parameter values determined by learning algorithms are stored and exposed as public attributes.

Non-proliferation of classes. Learning algorithms are the only objects to be represented using custom classes. Datasets are represented as NumPy arrays or SciPy sparse matrices. Hyper-parameter names and values are represented as standard Python strings or numbers whenever possible. This keeps scikitlearn easy to use and easy to combine with other libraries.

Composition. Many machine learning tasks are expressible as sequences or combinations of transformations to data. Some learning algorithms are also naturally viewed as meta-algorithms parametrized on other algorithms. Whenever feasible, such algorithms are implemented and composed from existing building blocks.

Sensible defaults. Whenever an operation requires a user-deﬁned parameter, an appropriate default value is deﬁned by the library. The default value should cause the operation to be performed in a sensible way (giving a baseline solution for the task at hand).


Basic Steps of Using Scikit-Learn API
Choose a class of model
Choose model hyperparameters
Arrage data into features matrix and target array
Fit model to data
Apply trained model to new data








