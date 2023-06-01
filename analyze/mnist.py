import tensorflow as tf
from analyze import analyze_apro
from analyze.utils import get_data


def log_results(
    epsilon,
    architecture,
    ei,
    trained_epsilon,
    test,
    metadata,
    dataset="mnist",
    batch_size=512,
    augmentation="standard",
    log_file="mnist.log",
):
    tf.config.run_functions_eagerly(True)
    g = analyze_apro(
        dataset=dataset,
        architecture=architecture,
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}/{dataset}_{architecture}_e{ei}.h5",
        metadata=metadata,
    )
    eval_result = g.evaluate(test)

    B_list = g.compute_bound()

    app_g = analyze_apro(
        dataset=dataset,
        architecture=f'approx_{architecture}.{{"B_list":{B_list}}}',
        epsilon=epsilon,
        batch_size=batch_size,
        loss="sparse_trades_ce.1.2",
        augmentation=augmentation,
        load_from=f"/pretrained_model/{dataset}/{dataset}_{architecture.lower()}_e{ei}.h5",
        metadata=metadata,
    )
    app_eval_result = app_g.evaluate(test)

    epsilon = app_g.epsilon
    tolerance = app_g.epsilon * app_g.sub_lipschitz
    total_error = app_g.compute_total_error(0)

    # Write results to log file
    with open(log_file, "a") as f:
        f.write(
            f"Architecture: {architecture}, Epsilon: {epsilon:.2f}, Trained Epsilon: {trained_epsilon}\n"
        )
        f.write(f"g_accuracy: {eval_result[1]:.4f}\n")
        f.write(f"g_vra: {eval_result[2]:.4f}\n")
        f.write(f"app_g_accuracy: {app_eval_result[1]:.4f}\n")
        f.write(f"app_g_vra: {app_eval_result[2]:.4f}\n")
        f.write(f"tolerance: {tolerance:.4f}\n")
        f.write(f"B_list: {B_list}\n")
        f.write(f"total error: {total_error:.4f}\n\n")


# List of epsilons and architectures
epsilons = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
]
trained_epsilons = [
    (0, 0.01),
    (1, 0.05),
]
architectures = ["cnn_2C2F"]

train, test, metadata = get_data("mnist", 512, "standard")

# Loop through each combination of epsilon and architecture and log the results
for architecture in architectures:
    for ei, trained_epsilon in trained_epsilons:
        for epsilon in epsilons:
            log_results(epsilon, architecture, ei, trained_epsilon, test, metadata)
