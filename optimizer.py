from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange

Task.init(project_name="PokemonClassification", task_name="OptHiperparTask", task_type=Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(
    base_task_id="71b225cac3054283bf1a6e76bd109b42",
    hyper_parameters=[
        UniformIntegerParameterRange('General/batch_size', min_value=16, max_value=64),
        UniformIntegerParameterRange('General/epochs', min_value=5, max_value=20),
        UniformIntegerParameterRange('General/img_height', min_value=64, max_value=256),
        UniformIntegerParameterRange('General/img_width', min_value=64, max_value=256),
    ],
    objective_metric_title="Overall Metrics",
    objective_metric_series="Validation Accuracy %",
    objective_metric_sign="max",
)

optimizer.start()
optimizer.wait()
optimizer.stop()
